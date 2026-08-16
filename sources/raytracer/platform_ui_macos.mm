#include "platform_ui.hxx"

#include "ui.hxx"

#import <AppKit/AppKit.h>

#include <cmath>
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
static NSToolbarItem* g_frame_scene_toolbar_item = nil;
static std::vector<std::string> g_recent_files = {};
static bool g_recent_files_initialized = false;
static std::vector<etx::Integrator*> g_integrators = {};
static NSVisualEffectView* g_startup_overlay = nil;
static NSProgressIndicator* g_startup_indicator = nil;
static NSTextField* g_startup_label = nil;
static NSToolbarItemGroup* g_render_toolbar_group = nil;
static NSToolbarItemGroup* g_panels_toolbar_group = nil;

static NSToolbarItemIdentifier const kOpenToolbarItem = @"com.etxtracer.toolbar.open";
static NSToolbarItemIdentifier const kSaveToolbarItem = @"com.etxtracer.toolbar.save";
static NSToolbarItemIdentifier const kRenderToolbarGroup = @"com.etxtracer.toolbar.render";
static NSToolbarItemIdentifier const kExportToolbarItem = @"com.etxtracer.toolbar.export";
static NSToolbarItemIdentifier const kFrameSceneToolbarItem = @"com.etxtracer.toolbar.frame-scene";
static NSToolbarItemIdentifier const kPanelsToolbarGroup = @"com.etxtracer.toolbar.panels";

@interface ETXPlatformUIController : NSObject<NSToolbarDelegate>
@property(nonatomic, assign) UI* ui;
- (void)performCommand:(id)sender;
- (void)performRenderCommand:(id)sender;
- (void)performPanelCommand:(id)sender;
@end

static ETXPlatformUIController* g_platform_ui_controller = nil;

static NSInteger selected_group_index(id sender) {
  if ([sender respondsToSelector:@selector(selectedIndex)]) {
    return [sender selectedIndex];
  }
  if ([sender respondsToSelector:@selector(selectedSegment)]) {
    return [sender selectedSegment];
  }
  return -1;
}

static NSString* ns_string(const char* value) {
  return value != nullptr ? [NSString stringWithUTF8String:value] : @"";
}

static NSImage* toolbar_symbol(NSString* name, NSString* accessibility_description) {
  constexpr CGFloat reference_point_size = 15.0;
  constexpr CGFloat target_image_area = 17.0 * 17.0;

  NSImage* system_image = [NSImage imageWithSystemSymbolName:name accessibilityDescription:accessibility_description];
  NSImageSymbolConfiguration* reference_configuration = [NSImageSymbolConfiguration
    configurationWithPointSize:reference_point_size weight:NSFontWeightRegular scale:NSImageSymbolScaleMedium];
  NSImage* reference_image = [system_image imageWithSymbolConfiguration:reference_configuration] ?: system_image;
  if (reference_image == nil) {
    return nil;
  }
  const CGFloat reference_area = reference_image.size.width * reference_image.size.height;
  if (reference_area <= 0.0) {
    return reference_image;
  }

  const CGFloat point_size = reference_point_size * std::sqrt(target_image_area / reference_area);
  NSImageSymbolConfiguration* configuration =
    [NSImageSymbolConfiguration configurationWithPointSize:point_size weight:NSFontWeightRegular scale:NSImageSymbolScaleMedium];
  NSImage* image = [system_image imageWithSymbolConfiguration:configuration] ?: reference_image;
  image.alignmentRect = NSMakeRect(0.0, 0.0, image.size.width, image.size.height);
  return image;
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

static NSMenuItem* add_command_item(NSMenu* menu, NSString* title, MenuCommand command, NSString* key_equivalent = @"", NSEventModifierFlags modifiers = NSEventModifierFlagCommand) {
  NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:title action:@selector(performCommand:) keyEquivalent:key_equivalent];
  item.target = g_platform_ui_controller;
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

- (void)performRenderCommand:(id)sender {
  if (self.ui == nullptr) {
    return;
  }

  constexpr MenuCommand commands[] = {
    MenuCommand::RunRenderer,
    MenuCommand::FinishRenderer,
    MenuCommand::StopRenderer,
    MenuCommand::RestartRenderer,
  };
  const NSInteger index = selected_group_index(sender);
  if ((index >= 0) && (index < static_cast<NSInteger>(sizeof(commands) / sizeof(commands[0])))) {
    self.ui->execute_menu_command(commands[index]);
  }
}

- (void)performPanelCommand:(id)sender {
  if (self.ui == nullptr) {
    return;
  }

  constexpr MenuCommand commands[] = {
    MenuCommand::ToggleSceneObjects,
    MenuCommand::ToggleProperties,
  };
  const NSInteger index = selected_group_index(sender);
  if ((index >= 0) && (index < static_cast<NSInteger>(sizeof(commands) / sizeof(commands[0])))) {
    self.ui->execute_menu_command(commands[index]);
  }
}

- (NSArray<NSToolbarItemIdentifier>*)toolbarAllowedItemIdentifiers:(NSToolbar*)toolbar {
  (void)toolbar;
  return @[
    kOpenToolbarItem, kSaveToolbarItem, kRenderToolbarGroup,
    kExportToolbarItem, kFrameSceneToolbarItem, kPanelsToolbarGroup,
    NSToolbarSpaceItemIdentifier, NSToolbarFlexibleSpaceItemIdentifier,
  ];
}

- (NSArray<NSToolbarItemIdentifier>*)toolbarDefaultItemIdentifiers:(NSToolbar*)toolbar {
  (void)toolbar;
  return @[
    kOpenToolbarItem, kSaveToolbarItem,
    NSToolbarSpaceItemIdentifier,
    kRenderToolbarGroup,
    NSToolbarFlexibleSpaceItemIdentifier,
    kFrameSceneToolbarItem, kExportToolbarItem,
    NSToolbarSpaceItemIdentifier,
    kPanelsToolbarGroup,
  ];
}

- (NSToolbarItem*)toolbar:(NSToolbar*)toolbar itemForItemIdentifier:(NSToolbarItemIdentifier)identifier willBeInsertedIntoToolbar:(BOOL)flag {
  (void)toolbar;
  (void)flag;

  if ([identifier isEqualToString:kRenderToolbarGroup]) {
    NSArray<NSImage*>* images = @[
      toolbar_symbol(@"play.circle", @"Start Rendering"),
      toolbar_symbol(@"checkmark.circle", @"Finish Current Iteration"),
      toolbar_symbol(@"stop.circle", @"Stop Rendering"),
      toolbar_symbol(@"arrow.clockwise.circle", @"Restart Rendering"),
    ];
    g_render_toolbar_group = [NSToolbarItemGroup groupWithItemIdentifier:kRenderToolbarGroup
      images:images
      selectionMode:NSToolbarItemGroupSelectionModeMomentary
      labels:@[@"Run", @"Finish", @"Stop", @"Restart"]
      target:self
      action:@selector(performRenderCommand:)];
    g_render_toolbar_group.label = @"Render";
    g_render_toolbar_group.paletteLabel = @"Render Controls";
    g_render_toolbar_group.toolTip = @"Rendering Controls";
    g_render_toolbar_group.controlRepresentation = NSToolbarItemGroupControlRepresentationExpanded;
    g_render_toolbar_group.autovalidates = NO;

    NSArray<NSToolbarItem*>* items = g_render_toolbar_group.subitems;
    items[0].toolTip = @"Start Rendering";
    items[1].toolTip = @"Finish Current Iteration";
    items[2].toolTip = @"Stop Rendering Immediately";
    items[3].toolTip = @"Restart Rendering";
    for (NSToolbarItem* item in items) {
      item.autovalidates = NO;
    }
    return g_render_toolbar_group;
  }

  if ([identifier isEqualToString:kPanelsToolbarGroup]) {
    NSArray<NSImage*>* images = @[
      toolbar_symbol(@"sidebar.left", @"Scene Objects"),
      toolbar_symbol(@"sidebar.right", @"Properties"),
    ];
    g_panels_toolbar_group = [NSToolbarItemGroup groupWithItemIdentifier:kPanelsToolbarGroup
      images:images
      selectionMode:NSToolbarItemGroupSelectionModeSelectAny
      labels:@[@"Objects", @"Properties"]
      target:self
      action:@selector(performPanelCommand:)];
    g_panels_toolbar_group.label = @"Panels";
    g_panels_toolbar_group.paletteLabel = @"Panels";
    g_panels_toolbar_group.toolTip = @"Show or Hide Panels";
    g_panels_toolbar_group.controlRepresentation = NSToolbarItemGroupControlRepresentationExpanded;
    return g_panels_toolbar_group;
  }

  NSString* label = nil;
  NSString* symbol = nil;
  NSString* help = nil;
  MenuCommand command = MenuCommand::OpenScene;
  if ([identifier isEqualToString:kOpenToolbarItem]) {
    label = @"Open";
    symbol = @"folder";
    help = @"Open Scene";
    command = MenuCommand::OpenScene;
  } else if ([identifier isEqualToString:kSaveToolbarItem]) {
    label = @"Save";
    symbol = @"tray.and.arrow.down";
    help = @"Save Scene";
    command = MenuCommand::SaveScene;
  } else if ([identifier isEqualToString:kExportToolbarItem]) {
    label = @"Export";
    symbol = @"arrow.up.forward.square";
    help = @"Export Current Image";
    command = MenuCommand::SaveImageRGB;
  } else if ([identifier isEqualToString:kFrameSceneToolbarItem]) {
    label = @"Frame";
    symbol = @"viewfinder";
    help = @"Frame Whole Scene";
    command = MenuCommand::ViewWholeScene;
  } else {
    return nil;
  }

  NSToolbarItem* item = [[NSToolbarItem alloc] initWithItemIdentifier:identifier];
  item.label = label;
  item.paletteLabel = label;
  item.toolTip = help;
  item.image = toolbar_symbol(symbol, help);
  item.target = self;
  item.action = @selector(performCommand:);
  item.tag = kCommandTagBase + static_cast<NSInteger>(command);
  if ([identifier isEqualToString:kFrameSceneToolbarItem]) {
    g_frame_scene_toolbar_item = item;
    item.autovalidates = NO;
  }
  return item;
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

  NSWindow* window = NSApp.keyWindow ?: NSApp.mainWindow;
  if (window != nil) {
    window.collectionBehavior =
      (window.collectionBehavior & ~NSWindowCollectionBehaviorFullScreenPrimary) | NSWindowCollectionBehaviorFullScreenNone;
  }

  [NSApp setWindowsMenu:nil];
  NSApp.mainMenu = main_menu;

  if (window != nil) {
    window.tabbingMode = NSWindowTabbingModeDisallowed;
    NSToolbar* toolbar = [[NSToolbar alloc] initWithIdentifier:@"com.etxtracer.toolbar.v2"];
    toolbar.delegate = g_platform_ui_controller;
    toolbar.displayMode = NSToolbarDisplayModeIconOnly;
    toolbar.allowsUserCustomization = YES;
    toolbar.autosavesConfiguration = YES;
    toolbar.centeredItemIdentifiers = [NSSet setWithObject:kRenderToolbarGroup];
    window.toolbarStyle = NSWindowToolbarStyleUnified;
    window.titleVisibility = NSWindowTitleVisible;
    window.toolbar = toolbar;
  }
  ui.set_embedded_menu_enabled(false);
  ui.set_embedded_toolbar_enabled(false);
}

void PlatformUI::update(UI& ui, const std::vector<std::string>& recent_files) {
  if (g_platform_ui_controller == nil) {
    return;
  }
  g_platform_ui_controller.ui = &ui;

  g_cpu_renderer_item.state = ui.current_renderer_mode() == RendererMode::CPURaytracing ? NSControlStateValueOn : NSControlStateValueOff;
  g_raster_renderer_item.state = ui.current_renderer_mode() == RendererMode::Rasterization ? NSControlStateValueOn : NSControlStateValueOff;
  g_gpu_renderer_item.state = ui.current_renderer_mode() == RendererMode::GPURaytracing ? NSControlStateValueOn : NSControlStateValueOff;
  g_gpu_renderer_item.enabled = ui.gpu_renderer_available();
  g_scene_objects_item.state = ui.scene_objects_visible() ? NSControlStateValueOn : NSControlStateValueOff;
  g_properties_item.state = ui.properties_visible() ? NSControlStateValueOn : NSControlStateValueOff;
  const BOOL frame_scene_enabled = ui.scene_view_commands_available() ? YES : NO;
  if (g_frame_scene_toolbar_item.enabled != frame_scene_enabled) {
    g_frame_scene_toolbar_item.enabled = frame_scene_enabled;
  }

  const RendererControlState& controls = ui.renderer_controls();
  NSArray<NSToolbarItem*>* render_items = g_render_toolbar_group.subitems;
  if (render_items.count == 4) {
    const BOOL enabled[] = {
      controls.can_run ? YES : NO,
      controls.can_finish ? YES : NO,
      controls.can_stop ? YES : NO,
      controls.can_restart ? YES : NO,
    };
    for (NSUInteger index = 0; index < render_items.count; ++index) {
      if (render_items[index].enabled != enabled[index]) {
        render_items[index].enabled = enabled[index];
      }
    }
  }
  if (g_panels_toolbar_group.subitems.count == 2) {
    [g_panels_toolbar_group setSelected:ui.scene_objects_visible() atIndex:0];
    [g_panels_toolbar_group setSelected:ui.properties_visible() atIndex:1];
    g_panels_toolbar_group.subitems[0].toolTip = ui.scene_objects_visible() ? @"Hide Scene Objects" : @"Show Scene Objects";
    g_panels_toolbar_group.subitems[1].toolTip = ui.properties_visible() ? @"Hide Properties" : @"Show Properties";
  }

  if (!g_recent_files_initialized || (g_recent_files != recent_files)) {
    g_recent_files_initialized = true;
    g_recent_files = recent_files;
    rebuild_recent_menu(recent_files);
  }
  std::vector<Integrator*> integrators(ui.integrator_count());
  for (uint64_t i = 0; i < ui.integrator_count(); ++i) {
    integrators[i] = ui.integrator(i);
  }
  if (g_integrators != integrators) {
    g_integrators = std::move(integrators);
    rebuild_integrator_menu(ui);
  } else {
    for (NSMenuItem* item in g_integrator_menu.itemArray) {
      if (![item.representedObject isKindOfClass:[NSNumber class]]) {
        continue;
      }
      const uint64_t index = [static_cast<NSNumber*>(item.representedObject) unsignedLongLongValue];
      Integrator* integrator = ui.integrator(index);
      item.enabled = (integrator != nullptr) && integrator->enabled();
      item.state = integrator == ui.current_integrator() ? NSControlStateValueOn : NSControlStateValueOff;
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
  g_cpu_renderer_item = nil;
  g_raster_renderer_item = nil;
  g_gpu_renderer_item = nil;
  g_scene_objects_item = nil;
  g_properties_item = nil;
  g_frame_scene_toolbar_item = nil;
  g_render_toolbar_group = nil;
  g_panels_toolbar_group = nil;
  g_recent_files.clear();
  g_recent_files_initialized = false;
  g_integrators.clear();
}

PlatformColorScheme PlatformUI::color_scheme() const {
  NSAppearanceName appearance = [NSApp.effectiveAppearance bestMatchFromAppearancesWithNames:@[NSAppearanceNameAqua, NSAppearanceNameDarkAqua]];
  return [appearance isEqualToString:NSAppearanceNameDarkAqua] ? PlatformColorScheme::Dark : PlatformColorScheme::Light;
}

bool PlatformUI::defers_initialization() const {
  return true;
}

}  // namespace etx
