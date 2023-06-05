use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassContents,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
            DebugUtilsMessengerCreateInfo,
        },
        Instance, InstanceCreateInfo, LayerProperties,
    },
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryAllocator,
        MemoryUsage, StandardMemoryAllocator,
    },
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{
        acquire_next_image, AcquireError, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainCreationError, SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{self, ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

#[derive(BufferContents, vulkano::pipeline::graphics::vertex_input::Vertex)]
#[repr(C)]
struct Vertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

struct Application {
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<SwapchainImage>>,
    memory_allocator: GenericMemoryAllocator<Arc<FreeListAllocator>>,
    vertex_buffer: Subbuffer<[Vertex]>,
    vert_shader: Arc<ShaderModule>,
    frag_shader: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    framebuffers: Vec<Arc<Framebuffer>>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    should_recreate_swapchain: bool,
}

impl Application {
    pub fn init() -> (Self, EventLoop<()>) {
        let (instance, debug_callback) = Self::create_instance();
        let (events_loop, surface) = Self::init_window(&instance);
        let (device, queue) = Self::choose_gpu(&instance, &surface);
        let (swapchain, images) = Self::create_swapchain(&device, &surface);
        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
        let vertex_buffer = Self::create_vertex_buffer(&memory_allocator);
        let (vert_shader, frag_shader) = Self::create_shaders(&device);
        let render_pass = Self::create_render_pass(&device, &swapchain);
        let pipeline = Self::create_pipeleine(&device, &render_pass, &vert_shader, &frag_shader);
        // Dynamic viewports allow us to recreate just the viewport when the window is resized.
        // Otherwise we would have to recreate the whole pipeline.
        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [0.0, 0.0],
            depth_range: 0.0..1.0,
        };
        let framebuffers = Self::recreate_framebuffers(&images, render_pass.clone(), &mut viewport);
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let should_recreate_swapchain = false;
        (
            Self {
                instance,
                surface,
                device,
                queue,
                swapchain,
                images,
                memory_allocator,
                vertex_buffer,
                vert_shader,
                frag_shader,
                render_pass,
                pipeline,
                viewport,
                framebuffers,
                command_buffer_allocator,
                should_recreate_swapchain,
            },
            events_loop,
        )
    }

    fn create_instance() -> (Arc<Instance>, Option<DebugUtilsMessenger>) {
        let library = VulkanLibrary::new().unwrap();
        let mut required_extensions = vulkano_win::required_extensions(&library);
        let layers = library.layer_properties().unwrap();
        println!("Supported layers:");
        layers.for_each(|l| println!("{}", l.name()));
        println!("No valid");
        let no_valid: Vec<LayerProperties> = library
            .layer_properties()
            .unwrap()
            .filter(|s| s.name() != "VK_LAYER_KHRONOS_validation")
            .collect();
        // .for_each(|s| println!("{}", s.name()));

        // TODO: handle validation layers in release builds
        let instance = {
            #[cfg(not(debug_assertions))]
            {
                Instance::new(
                    library,
                    InstanceCreateInfo {
                enabled_extensions: required_extensions,
                #[cfg(target_os = "macos")]
                // Enable enumerating devices that use non-conformant Vulkan implementations. (e.g.
                // MoltenVK)
                enumerate_portability: true,

                ..Default::default()
            },
                )
                .unwrap()
            }

            #[cfg(debug_assertions)]
            {
                required_extensions.ext_debug_utils = true;
                Instance::new(
                    library,
                    InstanceCreateInfo {
                enabled_extensions: required_extensions,
                // enabled_layers: no_valid.iter().map(|l| l.name().to_owned()).collect(),
                // enabled_layers: vec!["VK_LAYER_KHRONOS_validation".to_owned()],
                #[cfg(target_os = "macos")]
                // Enable enumerating devices that use non-conformant Vulkan implementations. (e.g.
                // MoltenVK)
                enumerate_portability: true,

                ..Default::default()
            },
                )
                .unwrap()
            }
        };
        let debug_callback = unsafe {
            let callback = DebugUtilsMessenger::new(
                instance.clone(),
                DebugUtilsMessengerCreateInfo {
                    message_severity: DebugUtilsMessageSeverity::ERROR
                        | DebugUtilsMessageSeverity::WARNING
                        | DebugUtilsMessageSeverity::INFO
                        | DebugUtilsMessageSeverity::VERBOSE,
                    message_type: DebugUtilsMessageType::GENERAL
                        | DebugUtilsMessageType::VALIDATION
                        | DebugUtilsMessageType::PERFORMANCE,
                    ..DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|msg| {
                        let severity = if msg.severity.intersects(DebugUtilsMessageSeverity::ERROR)
                        {
                            "error"
                        } else if msg.severity.intersects(DebugUtilsMessageSeverity::WARNING) {
                            "warning"
                        } else if msg.severity.intersects(DebugUtilsMessageSeverity::INFO) {
                            "information"
                        } else if msg.severity.intersects(DebugUtilsMessageSeverity::VERBOSE) {
                            "verbose"
                        } else {
                            panic!("no-impl");
                        };

                        let ty = if msg.ty.intersects(DebugUtilsMessageType::GENERAL) {
                            "general"
                        } else if msg.ty.intersects(DebugUtilsMessageType::VALIDATION) {
                            "validation"
                        } else if msg.ty.intersects(DebugUtilsMessageType::PERFORMANCE) {
                            "performance"
                        } else {
                            panic!("no-impl");
                        };

                        println!(
                            "{} {} {}: {}",
                            msg.layer_prefix.unwrap_or("unknown"),
                            ty,
                            severity,
                            msg.description
                        );
                    }))
                },
            )
            .ok();
            #[cfg(not(debug_assertions))]
            {
                let callback = None;
            }
            callback
        };
        (instance, debug_callback)
    }

    fn init_window(instance: &Arc<Instance>) -> (EventLoop<()>, Arc<Surface>) {
        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();
        (event_loop, surface)
    }

    fn choose_gpu(instance: &Arc<Instance>, surface: &Arc<Surface>) -> (Arc<Device>, Arc<Queue>) {
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                // Some devices may not support the extensions or features that your application, or
                // report properties and limits that are not sufficient for your application. These
                // should be filtered out here.
                p.supported_extensions().contains(&device_extensions)
            })
            .filter_map(|p| {
                // For each physical device, we try to find a suitable queue family that will execute
                // our draw commands.
                //
                // Devices can provide multiple queues to run commands in parallel (for example a draw
                // queue and a compute queue), similar to CPU threads. This is something you have to
                // have to manage manually in Vulkan. Queues of the same type belong to the same queue
                // family.
                //
                // Here, we look for a single queue family that is suitable for our purposes. In a
                // real-world application, you may want to use a separate dedicated transfer queue to
                // handle data transfers in parallel with graphics operations. You may also need a
                // separate queue for compute operations, if your application uses those.
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        // We select a queue family that supports graphics operations. When drawing to
                        // a window surface, as we do in this example, we also need to check that
                        // queues in this queue family are capable of presenting images to the surface.
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    // The code here searches for the first queue family that is suitable. If none is
                    // found, `None` is returned to `filter_map`, which disqualifies this physical
                    // device.
                    .map(|i| (p, i as u32))
            })
            // All the physical devices that pass the filters above are suitable for the application.
            // However, not every device is equal, some are preferred over others. Now, we assign each
            // physical device a score, and pick the device with the lowest ("best") score.
            //
            // In this example, we simply select the best-scoring device to use in the application.
            // In a real-world setting, you may want to use the best-scoring device only as a "default"
            // or "recommended" device, and let the user choose the device themself.
            .min_by_key(|(p, _)| {
                // We assign a lower score to device types that are likely to be faster/better.
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
            })
            .expect("no suitable physical device found");
        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );
        let (device, mut queues) = Device::new(
            // Which physical device to connect to.
            physical_device,
            DeviceCreateInfo {
                // A list of optional features and extensions that our program needs to work correctly.
                // Some parts of the Vulkan specs are optional and must be enabled manually at device
                // creation. In this example the only thing we are going to need is the `khr_swapchain`
                // extension that allows us to draw to a window.
                enabled_extensions: device_extensions,

                // The list of queues that we are going to use. Here we only use one queue, from the
                // previously chosen queue family.
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],

                ..Default::default()
            },
        )
        .unwrap();
        // Since we can request multiple queues, the `queues` variable is in fact an iterator. We only
        // use one queue in this example, so we just retrieve the first and only element of the
        // iterator.
        let queue = queues.next().unwrap();
        (device, queue)
    }

    fn create_swapchain(
        device: &Arc<Device>,
        surface: &Arc<Surface>,
    ) -> (Arc<Swapchain>, Vec<Arc<SwapchainImage>>) {
        // Before we can draw on the surface, we have to create what is called a swapchain. Creating a
        // swapchain allocates the color buffers that will contain the image that will ultimately be
        // visible on the screen. These images are returned alongside the swapchain.
        let (mut swapchain, images) = {
            // Querying the capabilities of the surface. When we create the swapchain we can only pass
            // values that are allowed by the capabilities.
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            // Choosing the internal format that the images will have.
            let image_format = Some(
                device
                    .physical_device()
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
                    .0,
            );
            let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

            // Please take a look at the docs for the meaning of the parameters we didn't mention.
            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count,

                    image_format,

                    // The dimensions of the window, only used to initially setup the swapchain.
                    //
                    // NOTE:
                    // On some drivers the swapchain dimensions are specified by
                    // `surface_capabilities.current_extent` and the swapchain size must use these
                    // dimensions. These dimensions are always the same as the window dimensions.
                    //
                    // However, other drivers don't specify a value, i.e.
                    // `surface_capabilities.current_extent` is `None`. These drivers will allow
                    // anything, but the only sensible value is the window dimensions.
                    //
                    // Both of these cases need the swapchain to use the window dimensions, so we just
                    // use that.
                    image_extent: window.inner_size().into(),

                    image_usage: ImageUsage::COLOR_ATTACHMENT,

                    // The alpha mode indicates how the alpha value of the final image will behave. For
                    // example, you can choose whether the window will be opaque or transparent.
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),

                    ..Default::default()
                },
            )
            .unwrap()
        };
        (swapchain, images)
    }

    fn create_vertex_buffer(
        memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
    ) -> Subbuffer<[Vertex]> {
        let vertices = [
            Vertex {
                position: [0.0, -1.0],
            },
            Vertex {
                position: [-1.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();
        vertex_buffer
    }

    fn create_shaders(device: &Arc<Device>) -> (Arc<ShaderModule>, Arc<ShaderModule>) {
        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: r"
                    #version 450

                    layout(location = 0) in vec2 position;

                    void main() {
                        gl_Position = vec4(position, 0.0, 1.0);
                    }
                ",
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: r"
                    #version 450

                    layout(location = 0) out vec4 f_color;

                    void main() {
                        f_color = vec4(1.0, 0.0, 0.0, 1.0);
                    }
                ",
            }
        }

        let vs = vs::load(device.clone()).unwrap();
        let fs = fs::load(device.clone()).unwrap();
        (vs, fs)
    }

    fn create_pipeleine(
        device: &Arc<Device>,
        render_pass: &Arc<RenderPass>,
        vs: &Arc<ShaderModule>,
        fs: &Arc<ShaderModule>,
    ) -> Arc<GraphicsPipeline> {
        // Before we draw we have to create what is called a pipeline. This is similar to an OpenGL
        // program, but much more specific.
        let pipeline = GraphicsPipeline::start()
            // We have to indicate which subpass of which render pass this pipeline is going to be used
            // in. The pipeline will only be usable from this particular subpass.
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            // We need to indicate the layout of the vertices.
            .vertex_input_state(
                <Vertex as vulkano::pipeline::graphics::vertex_input::Vertex>::per_vertex(),
            )
            // The content of the vertex buffer describes a list of triangles.
            .input_assembly_state(InputAssemblyState::new())
            // A Vulkan shader can in theory contain multiple entry points, so we have to specify
            // which one.
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            // Use a resizable viewport set to draw over the entire window
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            // See `vertex_shader`.
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
            .build(device.clone())
            .unwrap();
        pipeline
    }

    fn create_render_pass(device: &Arc<Device>, swapchain: &Arc<Swapchain>) -> Arc<RenderPass> {
        // The next step is to create a *render pass*, which is an object that describes where the
        // output of the graphics pipeline will go. It describes the layout of the images where the
        // colors, depth and/or stencil information will be written.
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    // `load: Clear` means that we ask the GPU to clear the content of this attachment
                    // at the start of the drawing.
                    load: Clear,
                    // `store: Store` means that we ask the GPU to store the output of the draw in the
                    // actual image. We could also ask it to discard the result.
                    store: Store,
                    // `format: <ty>` indicates the type of the format of the image. This has to be one
                    // of the types of the `vulkano::format` module (or alternatively one of your
                    // structs that implements the `FormatDesc` trait). Here we use the same format as
                    // the swapchain.
                    format: swapchain.image_format(),
                    // `samples: 1` means that we ask the GPU to use one sample to determine the value
                    // of each pixel in the color attachment. We could use a larger value
                    // (multisampling) for antialiasing. An example of this can be found in
                    // msaa-renderpass.rs.
                    samples: 1,
                },
            },
            pass: {
                // We use the attachment named `color` as the one and only color attachment.
                color: [color],
                // No depth-stencil attachment is indicated with empty brackets.
                depth_stencil: {},
            },
        )
        .unwrap();
        render_pass
    }

    fn recreate_framebuffers(
        images: &[Arc<SwapchainImage>],
        render_pass: Arc<RenderPass>,
        viewport: &mut Viewport,
    ) -> Vec<Arc<Framebuffer>> {
        let dimensions = images[0].dimensions().width_height();
        viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

        images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    }

    fn main_loop(mut self, events_loop: EventLoop<()>) {
        let mut previous_frame_end = Some(sync::now(self.device.clone()).boxed());
        events_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                self.should_recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
                // Do not draw the frame when the screen dimensions are zero. On Windows, this can
                // occur when minimizing the application.
                let window = self
                    .surface
                    .object()
                    .unwrap()
                    .downcast_ref::<Window>()
                    .unwrap();
                let dimensions = window.inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }

                // It is important to call this function from time to time, otherwise resources
                // will keep accumulating and you will eventually reach an out of memory error.
                // Calling this function polls various fences in order to determine what the GPU
                // has already processed, and frees the resources that are no longer needed.
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                // Whenever the window resizes we need to recreate everything dependent on the
                // window size. In this example that includes the swapchain, the framebuffers and
                // the dynamic state viewport.
                if self.should_recreate_swapchain {
                    // Use the new dimensions of the window.

                    let (new_swapchain, new_images) =
                        match self.swapchain.recreate(SwapchainCreateInfo {
                            image_extent: dimensions.into(),
                            ..self.swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            // This error tends to happen when the user is manually resizing the
                            // window. Simply restarting the loop is the easiest way to fix this
                            // issue.
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("failed to recreate swapchain: {e}"),
                        };

                    self.swapchain = new_swapchain;

                    // Because framebuffers contains a reference to the old swapchain, we need to
                    // recreate framebuffers as well.
                    self.framebuffers = Self::recreate_framebuffers(
                        &new_images,
                        self.render_pass.clone(),
                        &mut self.viewport,
                    );

                    self.should_recreate_swapchain = false;
                }

                // Before we can draw on the output, we have to *acquire* an image from the
                // swapchain. If no image is available (which happens if you submit draw commands
                // too quickly), then the function will block. This operation returns the index of
                // the image that we are allowed to draw upon.
                //
                // This function can block if no image is available. The parameter is an optional
                // timeout after which the function call will return an error.
                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(self.swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            self.should_recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                // `acquire_next_image` can be successful, but suboptimal. This means that the
                // swapchain image will still work, but it may not display correctly. With some
                // drivers this can be when the window resizes, but it may not cause the swapchain
                // to become out of date.
                if suboptimal {
                    self.should_recreate_swapchain = true;
                }

                // In order to draw, we have to build a *command buffer*. The command buffer object
                // holds the list of commands that are going to be executed.
                //
                // Building a command buffer is an expensive operation (usually a few hundred
                // microseconds), but it is known to be a hot path in the driver and is expected to
                // be optimized.
                //
                // Note that we have to pass a queue family when we create the command buffer. The
                // command buffer will only be executable on that given queue family.
                let mut builder = AutoCommandBufferBuilder::primary(
                    &self.command_buffer_allocator,
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    // Before we can draw, we have to *enter a render pass*.
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            // A list of values to clear the attachments with. This list contains
                            // one item for each attachment in the render pass. In this case, there
                            // is only one attachment, and we clear it with a blue color.
                            //
                            // Only attachments that have `LoadOp::Clear` are provided with clear
                            // values, any others should use `ClearValue::None` as the clear value.
                            clear_values: vec![Some([0.3, 0.1, 0.6, 1.0].into())],

                            ..RenderPassBeginInfo::framebuffer(
                                self.framebuffers[image_index as usize].clone(),
                            )
                        },
                        // The contents of the first (and only) subpass. This can be either
                        // `Inline` or `SecondaryCommandBuffers`. The latter is a bit more advanced
                        // and is not covered here.
                        SubpassContents::Inline,
                    )
                    .unwrap()
                    // We are now inside the first subpass of the render pass.
                    //
                    // TODO: Document state setting and how it affects subsequent draw commands.
                    .set_viewport(0, [self.viewport.clone()])
                    .bind_pipeline_graphics(self.pipeline.clone())
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())
                    // We add a draw command.
                    .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    // We leave the render pass. Note that if we had multiple subpasses we could
                    // have called `next_subpass` to jump to the next subpass.
                    .end_render_pass()
                    .unwrap();

                // Finish building the command buffer by calling `build`.
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    // The color output is now expected to contain our triangle. But in order to
                    // show it on the screen, we have to *present* the image by calling
                    // `then_swapchain_present`.
                    //
                    // This function does not actually present the image immediately. Instead it
                    // submits a present command at the end of the queue. This means that it will
                    // only be presented once the GPU has finished executing the command buffer
                    // that draws the triangle.
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            self.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        self.should_recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                        // previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => {}
        });
    }
}

fn main() {
    let (app, events_loop) = Application::init();
    app.main_loop(events_loop);
}
