#include "macos_menu.hxx"

#include "ui.hxx"

#import <AppKit/AppKit.h>

#include <filesystem>

using etx::MenuCommand;
using etx::UI;

static constexpr NSInteger kCommandTagBase = 1000;

static NSMenu* g_recent_menu = nil;
static NSMenu* g_integrator_menu = nil;
static NSMenuItem* g_cpu_renderer_item = nil;
static NSMenuItem* g_raster_renderer_item = nil;
static NSMenuItem* g_gpu_renderer_item = nil;
static NSMenuItem* g_scene_objects_item = nil;
static NSMenuItem* g_properties_item = nil;
static std::vector<std::string> g_recent_files = {};
static bool g_recent_files_initialized = false;
static uint64_t g_integrator_count = ~0ull;

@interface ETXMenuTarget : NSObject
@property(nonatomic, assign) UI* ui;
- (void)performCommand:(id)sender;
@end

static ETXMenuTarget* g_menu_target = nil;

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

static NSMenuItem* add_command_item(NSMenu* menu, NSString* title, MenuCommand command, NSString* key_equivalent = @"", NSEventModifierFlags modifiers = NSEventModifierFlagCommand) {
  NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:title action:@selector(performCommand:) keyEquivalent:key_equivalent];
  item.target = g_menu_target;
  item.tag = kCommandTagBase + static_cast<NSInteger>(command);
  item.keyEquivalentModifierMask = key_equivalent.length > 0 ? modifiers : 0;
  [menu addItem:item];
  return item;
}

static NSMenu* add_submenu(NSMenu* main_menu, NSString* title) {
  NSMenuItem* root_item = [[NSMenuItem alloc] initWithTitle:title action:nil keyEquivalent:@""];
  NSMenu* submenu = [[NSMenu alloc] initWithTitle:title];
  root_item.submenu = submenu;
  [main_menu addItem:root_item];
  return submenu;
}

static void rebuild_recent_menu(const std::vector<std::string>& recent_files) {
  [g_recent_menu removeAllItems];

  for (auto it = recent_files.rbegin(); it != recent_files.rend(); ++it) {
    const std::filesystem::path path(*it);
    const std::string display_name = path.filename().empty() ? *it : path.filename().string();
    NSMenuItem* item = add_command_item(g_recent_menu, ns_string(display_name.c_str()), MenuCommand::OpenRecentScene);
    item.representedObject = ns_string(it->c_str());
    item.toolTip = ns_string(it->c_str());
  }

  if (recent_files.empty()) {
    NSMenuItem* empty_item = [[NSMenuItem alloc] initWithTitle:@"No Recent Scenes" action:nil keyEquivalent:@""];
    empty_item.enabled = NO;
    [g_recent_menu addItem:empty_item];
  } else {
    [g_recent_menu addItem:[NSMenuItem separatorItem]];
    add_command_item(g_recent_menu, @"Clear Menu", MenuCommand::ClearRecentScenes);
  }
}

static void rebuild_integrator_menu(UI& ui) {
  [g_integrator_menu removeAllItems];
  for (uint64_t i = 0; i < ui.integrator_count(); ++i) {
    etx::Integrator* integrator = ui.integrator(i);
    if (integrator == nullptr) {
      continue;
    }
    NSMenuItem* item = add_command_item(g_integrator_menu, ns_string(integrator->name()), MenuCommand::SelectIntegrator);
    item.representedObject = @(i);
    item.enabled = integrator->enabled();
    item.state = integrator == ui.current_integrator() ? NSControlStateValueOn : NSControlStateValueOff;
  }
}

@implementation ETXMenuTarget

- (void)performCommand:(id)sender {
  if ((self.ui == nullptr) || (![sender isKindOfClass:[NSMenuItem class]])) {
    return;
  }

  NSMenuItem* item = static_cast<NSMenuItem*>(sender);
  const MenuCommand command = static_cast<MenuCommand>(item.tag - kCommandTagBase);
  uint32_t argument = 0u;
  std::string value = {};
  if ([item.representedObject isKindOfClass:[NSNumber class]]) {
    argument = [static_cast<NSNumber*>(item.representedObject) unsignedIntValue];
  } else if ([item.representedObject isKindOfClass:[NSString class]]) {
    value = [static_cast<NSString*>(item.representedObject) UTF8String];
  }
  self.ui->execute_menu_command(command, argument, value);
}

@end

namespace etx {

void setup_macos_menu(UI& ui) {
  g_menu_target = [[ETXMenuTarget alloc] init];
  g_menu_target.ui = &ui;

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
  add_command_item(application_menu, [@"Quit " stringByAppendingString:app_name], MenuCommand::Quit, @"q");

  NSMenu* file_menu = add_submenu(main_menu, @"File");
  add_command_item(file_menu, @"Open Scene…", MenuCommand::OpenScene, @"o");
  g_recent_menu = [[NSMenu alloc] initWithTitle:@"Open Recent"];
  NSMenuItem* recent_item = [[NSMenuItem alloc] initWithTitle:@"Open Recent" action:nil keyEquivalent:@""];
  recent_item.submenu = g_recent_menu;
  [file_menu addItem:recent_item];
  [file_menu addItem:[NSMenuItem separatorItem]];
  add_command_item(file_menu, @"Reload Scene", MenuCommand::ReloadScene, @"r");
  add_command_item(file_menu, @"Reload Geometry and Materials", MenuCommand::ReloadGeometry, @"g");
  [file_menu addItem:[NSMenuItem separatorItem]];
  add_command_item(file_menu, @"Save Scene", MenuCommand::SaveScene, @"s");
  add_command_item(file_menu, @"Save Scene As…", MenuCommand::SaveSceneAs, @"s", NSEventModifierFlagCommand | NSEventModifierFlagShift);

  NSMenu* edit_menu = add_submenu(main_menu, @"Edit");
  [edit_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Undo" action:@selector(undo:) keyEquivalent:@"z"]];
  [edit_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Redo" action:@selector(redo:) keyEquivalent:@"Z"]];
  [edit_menu addItem:[NSMenuItem separatorItem]];
  [edit_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"]];
  [edit_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"]];
  [edit_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"]];
  [edit_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Select All" action:@selector(selectAll:) keyEquivalent:@"a"]];

  NSMenu* renderer_menu = add_submenu(main_menu, @"Renderer");
  g_cpu_renderer_item = add_command_item(renderer_menu, @"CPU Raytracer", MenuCommand::SelectCPURenderer);
  g_raster_renderer_item = add_command_item(renderer_menu, @"Rasterizer", MenuCommand::SelectRasterRenderer);
  g_gpu_renderer_item = add_command_item(renderer_menu, @"GPU Raytracer", MenuCommand::SelectGPURenderer);

  g_integrator_menu = add_submenu(main_menu, @"Integrator");

  NSMenu* image_menu = add_submenu(main_menu, @"Image");
  add_command_item(image_menu, @"Open Reference Image…", MenuCommand::OpenReferenceImage, @"i");
  [image_menu addItem:[NSMenuItem separatorItem]];
  add_command_item(image_menu, @"Save Current Image (RGB)…", MenuCommand::SaveImageRGB, @"e");
  add_command_item(image_menu, @"Save Current Image (LDR)…", MenuCommand::SaveImageLDR, @"e", NSEventModifierFlagCommand | NSEventModifierFlagShift);
  add_command_item(image_menu, @"Use as Reference", MenuCommand::UseImageAsReference, @"r", NSEventModifierFlagCommand | NSEventModifierFlagShift);

  NSMenu* view_menu = add_submenu(main_menu, @"View");
  add_command_item(view_menu, @"View Whole Scene", MenuCommand::ViewWholeScene, @"0");
  NSMenu* direction_menu = [[NSMenu alloc] initWithTitle:@"View Scene"];
  NSMenuItem* direction_item = [[NSMenuItem alloc] initWithTitle:@"View Scene" action:nil keyEquivalent:@""];
  direction_item.submenu = direction_menu;
  [view_menu addItem:direction_item];
  add_command_item(direction_menu, @"From +X", MenuCommand::ViewPositiveX, @"1");
  add_command_item(direction_menu, @"From −X", MenuCommand::ViewNegativeX, @"2");
  add_command_item(direction_menu, @"From +Y", MenuCommand::ViewPositiveY, @"3");
  add_command_item(direction_menu, @"From −Y", MenuCommand::ViewNegativeY, @"4");
  add_command_item(direction_menu, @"From +Z", MenuCommand::ViewPositiveZ, @"5");
  add_command_item(direction_menu, @"From −Z", MenuCommand::ViewNegativeZ, @"6");
  [view_menu addItem:[NSMenuItem separatorItem]];
  add_command_item(view_menu, @"Increase Exposure", MenuCommand::IncreaseExposure, @"+");
  add_command_item(view_menu, @"Decrease Exposure", MenuCommand::DecreaseExposure, @"-");
  [view_menu addItem:[NSMenuItem separatorItem]];
  g_scene_objects_item = add_command_item(view_menu, @"Scene Objects", MenuCommand::ToggleSceneObjects, @"1", NSEventModifierFlagCommand | NSEventModifierFlagOption);
  g_properties_item = add_command_item(view_menu, @"Properties", MenuCommand::ToggleProperties, @"2", NSEventModifierFlagCommand | NSEventModifierFlagOption);

  NSMenu* window_menu = add_submenu(main_menu, @"Window");
  [window_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Minimize" action:@selector(performMiniaturize:) keyEquivalent:@"m"]];
  [window_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Zoom" action:@selector(performZoom:) keyEquivalent:@""]];
  [window_menu addItem:[NSMenuItem separatorItem]];
  [window_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Bring All to Front" action:@selector(arrangeInFront:) keyEquivalent:@""]];
  [NSApp setWindowsMenu:window_menu];

  NSApp.mainMenu = main_menu;
  ui.set_embedded_menu_enabled(false);
}

void update_macos_menu(UI& ui, const std::vector<std::string>& recent_files) {
  if (g_menu_target == nil) {
    return;
  }
  g_menu_target.ui = &ui;

  g_cpu_renderer_item.state = ui.current_renderer_mode() == RendererMode::CPURaytracing ? NSControlStateValueOn : NSControlStateValueOff;
  g_raster_renderer_item.state = ui.current_renderer_mode() == RendererMode::Rasterization ? NSControlStateValueOn : NSControlStateValueOff;
  g_gpu_renderer_item.state = ui.current_renderer_mode() == RendererMode::GPURaytracing ? NSControlStateValueOn : NSControlStateValueOff;
  g_gpu_renderer_item.enabled = ui.gpu_renderer_available();
  g_scene_objects_item.state = ui.scene_objects_visible() ? NSControlStateValueOn : NSControlStateValueOff;
  g_properties_item.state = ui.properties_visible() ? NSControlStateValueOn : NSControlStateValueOff;

  if (!g_recent_files_initialized || (g_recent_files != recent_files)) {
    g_recent_files_initialized = true;
    g_recent_files = recent_files;
    rebuild_recent_menu(recent_files);
  }
  if (g_integrator_count != ui.integrator_count()) {
    g_integrator_count = ui.integrator_count();
    rebuild_integrator_menu(ui);
  } else {
    for (uint64_t i = 0; i < ui.integrator_count(); ++i) {
      Integrator* integrator = ui.integrator(i);
      NSMenuItem* item = [g_integrator_menu itemAtIndex:static_cast<NSInteger>(i)];
      if ((integrator != nullptr) && (item != nil)) {
        item.enabled = integrator->enabled();
        item.state = integrator == ui.current_integrator() ? NSControlStateValueOn : NSControlStateValueOff;
      }
    }
  }
}

void shutdown_macos_menu() {
  if (g_menu_target != nil) {
    g_menu_target.ui = nullptr;
  }
  g_menu_target = nil;
  g_recent_menu = nil;
  g_integrator_menu = nil;
  g_cpu_renderer_item = nil;
  g_raster_renderer_item = nil;
  g_gpu_renderer_item = nil;
  g_scene_objects_item = nil;
  g_properties_item = nil;
  g_recent_files.clear();
  g_recent_files_initialized = false;
  g_integrator_count = ~0ull;
}

}  // namespace etx
