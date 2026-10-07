#include "platform_ui.hxx"

#include "ui.hxx"
#include <sokol_app.h>

#include <etx/core/core.hxx>
#include <etx/import/import_service.hxx>

#import <AppKit/AppKit.h>

using etx::MenuCommand;
using etx::UI;

static constexpr NSInteger kCommandTagBase = 1000;

static NSMenu* g_recent_menu = nil;
static NSMenu* g_integrator_menu = nil;
static NSMenuItem* g_scene_objects_item = nil;
static NSMenuItem* g_properties_item = nil;
static NSMenuItem* g_diagnostics_item = nil;
static std::vector<std::string> g_recent_files = {};
static bool g_recent_files_initialized = false;
static std::vector<etx::Integrator*> g_integrators = {};
static bool g_gpu_renderer_available = false;
static NSVisualEffectView* g_startup_overlay = nil;
static NSProgressIndicator* g_startup_indicator = nil;
static NSTextField* g_startup_label = nil;

@interface ETXPlatformUIController : NSObject <NSMenuItemValidation>
@property(nonatomic, assign) UI* ui;
- (void)performCommand:(id)sender;
- (BOOL)validateMenuItem:(NSMenuItem*)menu_item;
@end

static ETXPlatformUIController* g_platform_ui_controller = nil;

static NSString* ns_string(const char* value) {
  return value != nullptr ? [NSString stringWithUTF8String:value] : @"";
}

static NSString* application_name() {
  NSString* name = NSBundle.mainBundle.infoDictionary[@"CFBundleDisplayName"];
  if (name.length == 0) {
    name = NSBundle.mainBundle.infoDictionary[@"CFBundleName"];
  }
  return name.length > 0 ? name : NSProcessInfo.processInfo.processName;
}

static void install_application_icon() {
  NSURL* icon_url = [NSBundle.mainBundle URLForResource:@"AppIcon" withExtension:@"icns"];
  NSImage* icon = icon_url != nil ? [[NSImage alloc] initWithContentsOfURL:icon_url] : nil;
  if (icon != nil) {
    NSApp.applicationIconImage = icon;
  }
}

static NSMenuItem* add_command_item(NSMenu* menu, NSString* title, MenuCommand command) {
  const auto& info = UI::menu_command_info(command);
  NSString* key_equivalent = @"";
  if ((info.key >= 32) && (info.key <= 126)) {
    const unichar key = static_cast<unichar>(info.key);
    key_equivalent = [[NSString stringWithCharacters:&key length:1] lowercaseString];
  }
  NSEventModifierFlags modifiers = 0;
  if (info.modifiers & SAPP_MODIFIER_SUPER)
    modifiers |= NSEventModifierFlagCommand;
  if (info.modifiers & SAPP_MODIFIER_CTRL)
    modifiers |= NSEventModifierFlagControl;
  if (info.modifiers & SAPP_MODIFIER_ALT)
    modifiers |= NSEventModifierFlagOption;
  if (info.modifiers & SAPP_MODIFIER_SHIFT)
    modifiers |= NSEventModifierFlagShift;
  NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:title action:@selector(performCommand:) keyEquivalent:key_equivalent];
  item.target = g_platform_ui_controller;
  item.tag = kCommandTagBase + static_cast<NSInteger>(command);
  item.keyEquivalentModifierMask = modifiers;
  [menu addItem:item];
  return item;
}

static NSMenuItem* add_command_item(NSMenu* menu, MenuCommand command) {
  NSString* title = [ns_string(UI::menu_command_info(command).label) stringByReplacingOccurrencesOfString:@"..." withString:@"\u2026"];
  return add_command_item(menu, title, command);
}

static NSMenu* add_submenu(NSMenu* main_menu, NSString* title) {
  NSMenuItem* root_item = [[NSMenuItem alloc] initWithTitle:title action:nil keyEquivalent:@""];
  NSMenu* submenu = [[NSMenu alloc] initWithTitle:title];
  root_item.submenu = submenu;
  [main_menu addItem:root_item];
  return submenu;
}

static void add_import_menu(NSMenu* menu, MenuCommand command) {
  NSMenu* submenu = add_submenu(menu, ns_string(UI::menu_command_info(command).label));
  for (const auto& importer : etx::import_service().importers()) {
    const std::string label = importer.name + " (" + importer.extensions + ")\u2026";
    NSMenuItem* item = add_command_item(submenu, ns_string(label.c_str()), command);
    item.representedObject = ns_string(importer.id.c_str());
  }
}

static void rebuild_recent_menu(const std::vector<std::string>& recent_files) {
  [g_recent_menu removeAllItems];

  for (auto it = recent_files.rbegin(); it != recent_files.rend(); ++it) {
    const std::string display_name = etx::utf8_file_name(*it);
    NSMenuItem* item = add_command_item(g_recent_menu, ns_string(display_name.c_str()), MenuCommand::OpenRecentScene);
    item.representedObject = ns_string(it->c_str());
    item.toolTip = ns_string(it->c_str());
  }

  if (recent_files.empty()) {
    NSMenuItem* empty_item = [[NSMenuItem alloc] initWithTitle:@"No Recent Files" action:nil keyEquivalent:@""];
    empty_item.enabled = NO;
    [g_recent_menu addItem:empty_item];
  } else {
    [g_recent_menu addItem:[NSMenuItem separatorItem]];
    add_command_item(g_recent_menu, MenuCommand::ClearRecentScenes);
  }
}

static void rebuild_integrator_menu(UI& ui) {
  [g_integrator_menu removeAllItems];

  const uint32_t raster_argument = UI::render_configuration_argument(etx::RendererMode::Rasterization, kInvalidIndex);
  NSMenuItem* raster_item = add_command_item(g_integrator_menu, @"Raster Preview", MenuCommand::SelectIntegrator);
  raster_item.representedObject = @(raster_argument);
  raster_item.state = ui.render_configuration_selected(raster_argument) ? NSControlStateValueOn : NSControlStateValueOff;

  [g_integrator_menu addItem:[NSMenuItem separatorItem]];
  NSMenuItem* cpu_header = [[NSMenuItem alloc] initWithTitle:@"CPU" action:nil keyEquivalent:@""];
  cpu_header.enabled = NO;
  [g_integrator_menu addItem:cpu_header];
  for (uint64_t i = 0; i < ui.integrator_count(); ++i) {
    etx::Integrator* integrator = ui.integrator(i);
    if ((integrator == nullptr) || (integrator->enabled() == false)) {
      continue;
    }
    const uint32_t argument = UI::render_configuration_argument(etx::RendererMode::CPURaytracing, static_cast<uint32_t>(i));
    const std::string label = UI::render_configuration_label(etx::RendererMode::CPURaytracing, integrator);
    NSMenuItem* item = add_command_item(g_integrator_menu, ns_string(label.c_str()), MenuCommand::SelectIntegrator);
    item.representedObject = @(argument);
    item.state = ui.render_configuration_selected(argument) ? NSControlStateValueOn : NSControlStateValueOff;
  }

  if (ui.gpu_renderer_available()) {
    [g_integrator_menu addItem:[NSMenuItem separatorItem]];
    NSMenuItem* gpu_header = [[NSMenuItem alloc] initWithTitle:@"GPU" action:nil keyEquivalent:@""];
    gpu_header.enabled = NO;
    [g_integrator_menu addItem:gpu_header];
    for (uint64_t i = 0; i < ui.integrator_count(); ++i) {
      etx::Integrator* integrator = ui.integrator(i);
      if ((integrator == nullptr) || (integrator->enabled() == false) || (UI::gpu_integrator_supported(integrator->type()) == false)) {
        continue;
      }
      const uint32_t argument = UI::render_configuration_argument(etx::RendererMode::GPURaytracing, static_cast<uint32_t>(i));
      const std::string label = UI::render_configuration_label(etx::RendererMode::GPURaytracing, integrator);
      NSMenuItem* item = add_command_item(g_integrator_menu, ns_string(label.c_str()), MenuCommand::SelectIntegrator);
      item.representedObject = @(argument);
      item.state = ui.render_configuration_selected(argument) ? NSControlStateValueOn : NSControlStateValueOff;
    }
  }
}

@implementation ETXPlatformUIController

- (void)performCommand:(id)sender {
  if ((self.ui == nullptr) || (![sender respondsToSelector:@selector(tag)])) {
    return;
  }

  const MenuCommand command = static_cast<MenuCommand>([sender tag] - kCommandTagBase);
  uint32_t argument = 0u;
  std::string value = {};
  if ([sender isKindOfClass:[NSMenuItem class]]) {
    NSMenuItem* item = static_cast<NSMenuItem*>(sender);
    if ([item.representedObject isKindOfClass:[NSNumber class]]) {
      argument = [static_cast<NSNumber*>(item.representedObject) unsignedIntValue];
    } else if ([item.representedObject isKindOfClass:[NSString class]]) {
      value = [static_cast<NSString*>(item.representedObject) UTF8String];
    }
  }
  self.ui->execute_menu_command(command, argument, value);
}

- (BOOL)validateMenuItem:(NSMenuItem*)menu_item {
  if (self.ui == nullptr) {
    return NO;
  }

  if ((menu_item.tag < kCommandTagBase) || (menu_item.tag >= (kCommandTagBase + static_cast<NSInteger>(MenuCommand::Count))))
    return NO;
  const MenuCommand command = static_cast<MenuCommand>(menu_item.tag - kCommandTagBase);
  return self.ui->menu_command_available(command) ? YES : NO;
}

@end

namespace etx {

PlatformUI& platform_ui() {
  static PlatformUI instance = {};
  return instance;
}

void PlatformUI::prepare_application() {
  [NSWindow setAllowsAutomaticWindowTabbing:NO];
}

void PlatformUI::show_startup() {
  install_application_icon();

  NSWindow* window = NSApp.keyWindow ?: NSApp.mainWindow;
  NSView* content_view = window.contentView;
  if ((content_view == nil) || (g_startup_overlay != nil)) {
    return;
  }

  g_startup_overlay = [[NSVisualEffectView alloc] initWithFrame:content_view.bounds];
  g_startup_overlay.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
  g_startup_overlay.blendingMode = NSVisualEffectBlendingModeWithinWindow;
  g_startup_overlay.material = NSVisualEffectMaterialUnderWindowBackground;
  g_startup_overlay.state = NSVisualEffectStateActive;

  g_startup_indicator = [[NSProgressIndicator alloc] initWithFrame:NSZeroRect];
  g_startup_indicator.style = NSProgressIndicatorStyleSpinning;
  g_startup_indicator.controlSize = NSControlSizeRegular;
  g_startup_indicator.translatesAutoresizingMaskIntoConstraints = NO;
  [g_startup_indicator startAnimation:nil];

  g_startup_label = [NSTextField labelWithString:@"Preparing ETX Tracer…\nThe first launch and large scenes may take a moment."];
  g_startup_label.alignment = NSTextAlignmentCenter;
  g_startup_label.font = [NSFont systemFontOfSize:15.0 weight:NSFontWeightMedium];
  g_startup_label.maximumNumberOfLines = 2;
  g_startup_label.translatesAutoresizingMaskIntoConstraints = NO;

  [g_startup_overlay addSubview:g_startup_indicator];
  [g_startup_overlay addSubview:g_startup_label];
  [NSLayoutConstraint activateConstraints:@[
    [g_startup_indicator.centerXAnchor constraintEqualToAnchor:g_startup_overlay.centerXAnchor],
    [g_startup_indicator.centerYAnchor constraintEqualToAnchor:g_startup_overlay.centerYAnchor constant:-24.0],
    [g_startup_label.topAnchor constraintEqualToAnchor:g_startup_indicator.bottomAnchor constant:16.0],
    [g_startup_label.centerXAnchor constraintEqualToAnchor:g_startup_overlay.centerXAnchor],
    [g_startup_label.leadingAnchor constraintGreaterThanOrEqualToAnchor:g_startup_overlay.leadingAnchor constant:24.0],
    [g_startup_label.trailingAnchor constraintLessThanOrEqualToAnchor:g_startup_overlay.trailingAnchor constant:-24.0],
  ]];

  [content_view addSubview:g_startup_overlay positioned:NSWindowAbove relativeTo:nil];
  [g_startup_overlay displayIfNeeded];
}

void PlatformUI::finish_startup(bool succeeded) {
  if (g_startup_overlay == nil) {
    return;
  }
  if (succeeded) {
    [g_startup_indicator stopAnimation:nil];
    [g_startup_overlay removeFromSuperview];
    g_startup_indicator = nil;
    g_startup_label = nil;
    g_startup_overlay = nil;
  } else {
    [g_startup_indicator stopAnimation:nil];
    g_startup_indicator.hidden = YES;
    g_startup_label.stringValue = @"ETX Tracer could not initialize the rendering system.";
  }
}

void PlatformUI::setup(UI& ui) {
  g_platform_ui_controller = [[ETXPlatformUIController alloc] init];
  g_platform_ui_controller.ui = &ui;

  NSMenu* main_menu = [[NSMenu alloc] initWithTitle:@"Main Menu"];
  NSString* app_name = application_name();

  NSMenu* application_menu = add_submenu(main_menu, app_name);
  NSMenuItem* about_item = [[NSMenuItem alloc] initWithTitle:[@"About " stringByAppendingString:app_name] action:@selector(orderFrontStandardAboutPanel:) keyEquivalent:@""];
  [application_menu addItem:about_item];
  [application_menu addItem:[NSMenuItem separatorItem]];
  NSMenu* services_menu = [[NSMenu alloc] initWithTitle:@"Services"];
  NSMenuItem* services_item = [[NSMenuItem alloc] initWithTitle:@"Services" action:nil keyEquivalent:@""];
  services_item.submenu = services_menu;
  [application_menu addItem:services_item];
  NSApp.servicesMenu = services_menu;
  [application_menu addItem:[NSMenuItem separatorItem]];
  NSMenuItem* hide_item = [[NSMenuItem alloc] initWithTitle:[@"Hide " stringByAppendingString:app_name] action:@selector(hide:) keyEquivalent:@"h"];
  [application_menu addItem:hide_item];
  NSMenuItem* hide_others_item = [[NSMenuItem alloc] initWithTitle:@"Hide Others" action:@selector(hideOtherApplications:) keyEquivalent:@"h"];
  hide_others_item.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagOption;
  [application_menu addItem:hide_others_item];
  [application_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Show All" action:@selector(unhideAllApplications:) keyEquivalent:@""]];
  [application_menu addItem:[NSMenuItem separatorItem]];
  add_command_item(application_menu, [@"Quit " stringByAppendingString:app_name], MenuCommand::Quit);

  NSMenu* file_menu = add_submenu(main_menu, @"File");
  add_command_item(file_menu, MenuCommand::OpenScene);
  g_recent_menu = add_submenu(file_menu, ns_string(UI::menu_command_info(MenuCommand::OpenRecentScene).label));
  [file_menu addItem:[NSMenuItem separatorItem]];
  add_command_item(file_menu, MenuCommand::SaveScene);
  add_command_item(file_menu, MenuCommand::SaveSceneAs);
  [file_menu addItem:[NSMenuItem separatorItem]];
  add_command_item(file_menu, MenuCommand::AddNativeScene);
  add_import_menu(file_menu, MenuCommand::ImportScene);
  add_import_menu(file_menu, MenuCommand::ImportIntoScene);
  add_command_item(file_menu, MenuCommand::ConvertScene);
  [file_menu addItem:[NSMenuItem separatorItem]];
  add_command_item(file_menu, MenuCommand::ReloadScene);
  add_command_item(file_menu, MenuCommand::ReloadGeometry);

  NSMenu* edit_menu = add_submenu(main_menu, @"Edit");
  [edit_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Undo" action:@selector(undo:) keyEquivalent:@"z"]];
  [edit_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Redo" action:@selector(redo:) keyEquivalent:@"Z"]];
  [edit_menu addItem:[NSMenuItem separatorItem]];
  [edit_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"]];
  [edit_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"]];
  [edit_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"]];
  [edit_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Select All" action:@selector(selectAll:) keyEquivalent:@"a"]];

  NSMenu* view_menu = add_submenu(main_menu, @"View");
  add_command_item(view_menu, MenuCommand::ViewWholeScene);
  NSMenu* direction_menu = add_submenu(view_menu, @"Camera Views");
  add_command_item(direction_menu, MenuCommand::ViewPositiveX);
  add_command_item(direction_menu, MenuCommand::ViewNegativeX);
  add_command_item(direction_menu, MenuCommand::ViewPositiveY);
  add_command_item(direction_menu, MenuCommand::ViewNegativeY);
  add_command_item(direction_menu, MenuCommand::ViewPositiveZ);
  add_command_item(direction_menu, MenuCommand::ViewNegativeZ);
  [view_menu addItem:[NSMenuItem separatorItem]];
  NSMenu* exposure_menu = add_submenu(view_menu, @"Exposure");
  add_command_item(exposure_menu, MenuCommand::IncreaseExposure);
  add_command_item(exposure_menu, MenuCommand::DecreaseExposure);
  [view_menu addItem:[NSMenuItem separatorItem]];
  NSMenu* panels_menu = add_submenu(view_menu, @"Panels");
  g_scene_objects_item = add_command_item(panels_menu, MenuCommand::ToggleSceneObjects);
  g_properties_item = add_command_item(panels_menu, MenuCommand::ToggleProperties);
  g_diagnostics_item = add_command_item(panels_menu, MenuCommand::ToggleMemoryDiagnostics);
  add_command_item(view_menu, MenuCommand::ResetLayout);

  NSMenu* render_menu = add_submenu(main_menu, @"Render");
  add_command_item(render_menu, MenuCommand::RunRenderer);
  add_command_item(render_menu, MenuCommand::FinishRenderer);
  add_command_item(render_menu, MenuCommand::StopRenderer);
  add_command_item(render_menu, MenuCommand::RestartRenderer);
  [render_menu addItem:[NSMenuItem separatorItem]];
  g_integrator_menu = add_submenu(render_menu, ns_string(UI::menu_command_info(MenuCommand::SelectIntegrator).label));

  NSMenu* image_menu = add_submenu(main_menu, @"Image");
  add_command_item(image_menu, MenuCommand::SaveImageRGB);
  add_command_item(image_menu, MenuCommand::SaveImageLDR);
  [image_menu addItem:[NSMenuItem separatorItem]];
  NSMenu* reference_menu = add_submenu(image_menu, @"Reference Image");
  add_command_item(reference_menu, MenuCommand::OpenReferenceImage);
  add_command_item(reference_menu, MenuCommand::UseImageAsReference);

  NSMenu* tools_menu = add_submenu(main_menu, @"Tools");
  add_command_item(tools_menu, MenuCommand::ImportPlugins);
  NSMenu* help_menu = add_submenu(main_menu, @"Help");
  add_command_item(help_menu, MenuCommand::KeyboardShortcuts);
  NSApp.helpMenu = help_menu;

  NSWindow* window = NSApp.keyWindow ?: NSApp.mainWindow;
  if (window != nil) {
    window.collectionBehavior = (window.collectionBehavior & ~NSWindowCollectionBehaviorFullScreenPrimary) | NSWindowCollectionBehaviorFullScreenNone;
    window.tabbingMode = NSWindowTabbingModeDisallowed;
    window.toolbar = nil;
  }

  [NSApp setWindowsMenu:nil];
  NSApp.mainMenu = main_menu;

  ui.set_embedded_menu_enabled(false);
  ui.set_embedded_toolbar_enabled(true);
}

void PlatformUI::update(UI& ui, const std::vector<std::string>& recent_files) {
  if (g_platform_ui_controller == nil) {
    return;
  }
  g_platform_ui_controller.ui = &ui;

  g_scene_objects_item.state = ui.scene_objects_visible() ? NSControlStateValueOn : NSControlStateValueOff;
  g_properties_item.state = ui.properties_visible() ? NSControlStateValueOn : NSControlStateValueOff;
  g_diagnostics_item.state = ui.diagnostics_visible() ? NSControlStateValueOn : NSControlStateValueOff;

  if (!g_recent_files_initialized || (g_recent_files != recent_files)) {
    g_recent_files_initialized = true;
    g_recent_files = recent_files;
    rebuild_recent_menu(recent_files);
  }
  std::vector<Integrator*> integrators(ui.integrator_count());
  for (uint64_t i = 0; i < ui.integrator_count(); ++i) {
    integrators[i] = ui.integrator(i);
  }
  if ((g_integrators != integrators) || (g_gpu_renderer_available != ui.gpu_renderer_available())) {
    g_integrators = std::move(integrators);
    g_gpu_renderer_available = ui.gpu_renderer_available();
    rebuild_integrator_menu(ui);
  } else {
    for (NSMenuItem* item in g_integrator_menu.itemArray) {
      if (![item.representedObject isKindOfClass:[NSNumber class]]) {
        continue;
      }
      const uint32_t argument = [static_cast<NSNumber*>(item.representedObject) unsignedIntValue];
      item.state = ui.render_configuration_selected(argument) ? NSControlStateValueOn : NSControlStateValueOff;
    }
  }
}

void PlatformUI::shutdown() {
  [g_startup_indicator stopAnimation:nil];
  [g_startup_overlay removeFromSuperview];
  g_startup_indicator = nil;
  g_startup_label = nil;
  g_startup_overlay = nil;
  if (g_platform_ui_controller != nil) {
    g_platform_ui_controller.ui = nullptr;
  }
  g_platform_ui_controller = nil;
  g_recent_menu = nil;
  g_integrator_menu = nil;
  g_scene_objects_item = nil;
  g_properties_item = nil;
  g_diagnostics_item = nil;
  g_recent_files.clear();
  g_recent_files_initialized = false;
  g_integrators.clear();
  g_gpu_renderer_available = false;
}

PlatformColorScheme PlatformUI::color_scheme() const {
  NSAppearanceName appearance = [NSApp.effectiveAppearance bestMatchFromAppearancesWithNames:@[ NSAppearanceNameAqua, NSAppearanceNameDarkAqua ]];
  return [appearance isEqualToString:NSAppearanceNameDarkAqua] ? PlatformColorScheme::Dark : PlatformColorScheme::Light;
}

bool PlatformUI::defers_initialization() const {
  return true;
}

}  // namespace etx
