import bpy
import os.path
import math
import sys

C = bpy.context
D = bpy.data
scene = D.scenes['Scene']

# NOTE: Modify this line to your BG image path
background = '/home/yulin/Desktop/background.jpg'

# cameras: a list of camera positions
# a camera position is defined by two parameters: (theta, phi),
# where we fix the "r" of (r, theta, phi) in spherical coordinate system.

# 5 orientations: front, right, back, left, top
# cameras = [
#    (60, 0), (60, 90), (60, 180), (60, 270),
#    (0, 0)
#]

# 12 orientations around the object with 30-deg elevation
cameras = [(60, i) for i in range(0, 360, 30)]

render_setting = scene.render

# output image size = (W, H)
w = 600
h = 600
render_setting.resolution_x = w
render_setting.resolution_y = h


def main():
    argv = sys.argv
    argv = argv[argv.index('--') + 1:]

    if len(argv) != 2:
        print('phong.py args: <3d mesh path> <image dir>')
        exit(-1)

    model = argv[0]
    image_dir = argv[1]

    # blender has no native support for off files
    # install_off_addon()

    init_camera()
    fix_camera_to_origin()

    do_model(model, image_dir)


def install_off_addon():
    try:
        bpy.ops.wm.addon_install(
            overwrite=False,
            filepath=os.path.dirname(__file__) +
            '/blender-off-addon/import_off.py'
        )
        bpy.ops.wm.addon_enable(module='import_off')
    except Exception:
        print("""Import blender-off-addon failed.
              Did you pull the blender-off-addon submodule?
              $ git submodule update --recursive --remote
              """)
        exit(-1)


def init_camera():
    cam = D.objects['Camera']
    # select the camera object
    scene.objects.active = cam
    cam.select = True

    # set the rendering mode to orthogonal and scale
    C.object.data.type = 'ORTHO'
    C.object.data.ortho_scale = 2.


def fix_camera_to_origin():
    origin_name = 'Origin'

    # create origin
    try:
        origin = D.objects[origin_name]
    except KeyError:
        bpy.ops.object.empty_add(type='SPHERE')
        D.objects['Empty'].name = origin_name
        origin = D.objects[origin_name]

    origin.location = (0, 0, 0)

    cam = D.objects['Camera']
    scene.objects.active = cam
    cam.select = True

    if 'Track To' not in cam.constraints:
        bpy.ops.object.constraint_add(type='TRACK_TO')

    cam.constraints['Track To'].target = origin
    cam.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
    cam.constraints['Track To'].up_axis = 'UP_Y'


def do_model(path, image_dir):
    name = load_model(path)
    # Remove following 2 lines because ShapeNetCoreV2 is already normalized
    # center_model(name)
    # normalize_model(name)
    # image_subdir = os.path.join(image_dir, name)
    image_subdir = image_dir
    for i, c in enumerate(cameras):
        move_camera(c)
        render()
        save(image_subdir, '%s.%d' % (name, i))

    delete_model(name)


def load_model(path):
    d = os.path.dirname(path)
    ext = path.split('.')[-1]

    name = os.path.basename(path).split('.')[0]
    # handle weird object naming by Blender for stl files
    if ext == 'stl':
        name = name.title().replace('_', ' ')

    if name not in D.objects:
        print('loading :' + name)
        if ext == 'stl':
            bpy.ops.import_mesh.stl(filepath=path, directory=d,
                                    filter_glob='*.stl')
        elif ext == 'off':
            bpy.ops.import_mesh.off(filepath=path, filter_glob='*.off')
        elif ext == 'obj':
            bpy.ops.import_scene.obj(filepath=path, filter_glob='*.obj')
        else:
            print('Currently .{} file type is not supported.'.format(ext))
            exit(-1)
    return name


def delete_model(name):
    for ob in scene.objects:
        if ob.type == 'MESH' and ob.name.startswith(name):
            ob.select = True
        else:
            ob.select = False
    bpy.ops.object.delete()


def center_model(name):
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
    for obj in D.objects:
        print(obj.name)
    D.objects[name].location = (0, 0, 0)


def normalize_model(name):
    obj = D.objects[name]
    dim = obj.dimensions
    print('original dim:' + str(dim))
    if max(dim) > 0:
        dim = dim / max(dim)
    obj.dimensions = dim

    print('new dim:' + str(dim))


def move_camera(coord):
    def deg2rad(deg):
        return deg * math.pi / 180.

    r = 3.
    theta, phi = deg2rad(coord[0]), deg2rad(coord[1])
    loc_x = r * math.sin(theta) * math.cos(phi)
    loc_y = r * math.sin(theta) * math.sin(phi)
    loc_z = r * math.cos(theta)

    D.objects['Camera'].location = (loc_x, loc_y, loc_z)

    image_node = bpy.context.scene.node_tree.nodes[0]
    image_node.image = bpy.data.images.load(background)
    file_output_node = bpy.context.scene.node_tree.nodes[4]
    file_output_node.file_slots[0].path = 'blender-######.color.png' # blender placeholder #


def render():
    bpy.ops.render.render()

def node_setting_init():
    """node settings for render rgb images
    mainly for compositing the background images
    """

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)
    
    image_node = tree.nodes.new('CompositorNodeImage')
    scale_node = tree.nodes.new('CompositorNodeScale')
    alpha_over_node = tree.nodes.new('CompositorNodeAlphaOver')
    render_layer_node = tree.nodes.new('CompositorNodeRLayers')
    file_output_node = tree.nodes.new('CompositorNodeOutputFile')

    scale_node.space = 'RENDER_SIZE'
    #file_output_node.base_path = g_syn_rgb_folder

    links.new(image_node.outputs[0], scale_node.inputs[0])
    links.new(scale_node.outputs[0], alpha_over_node.inputs[1])
    links.new(render_layer_node.outputs[0], alpha_over_node.inputs[2])
    links.new(alpha_over_node.outputs[0], file_output_node.inputs[0])

def scene_setting_init():
    """initialize blender setting configurations
    """
    sce = bpy.context.scene.name
    bpy.data.scenes[sce].render.engine = 'CYCLES'
    bpy.data.scenes[sce].cycles.film_transparent = True

    #output
    bpy.data.scenes[sce].render.image_settings.color_mode = 'RGB'
    bpy.data.scenes[sce].render.image_settings.color_depth = '16'
    bpy.data.scenes[sce].render.image_settings.file_format = 'PNG'

    #dimensions
    #bpy.data.scenes[sce].render.resolution_x = g_resolution_x
    #bpy.data.scenes[sce].render.resolution_y = g_resolution_y
    #bpy.data.scenes[sce].render.resolution_percentage = g_resolution_percentage


def save(image_dir, name):
    path = os.path.join(image_dir, name + '.png')
    D.images['Render Result'].save_render(filepath=path)
    print('save to ' + path)


if __name__ == '__main__':
    node_setting_init()
    scene_setting_init()
    main()
