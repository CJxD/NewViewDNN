import bpy
import os
import sys
import numpy as np

from mathutils import Vector

context = bpy.context
camera_size = Vector((128, 128))
texture_size = Vector((128, 128))
lerp_interval = 0.1
cam_pos_A = Vector((-1.0, 1.0, 0.2))
cam_rot_A = Vector((1.431, 0.0, -2.356))
cam_pos_B = Vector((-1.0, -1.0, 0.2))
cam_rot_B = Vector((1.431, 0.0, -0.785))

def reset_scene():
	bpy.ops.wm.read_factory_settings()

	for scene in bpy.data.scenes:
		for obj in scene.objects:
			scene.objects.unlink(obj)

	# only worry about data in the startup scene
	for bpy_data_iter in (
			bpy.data.objects,
			bpy.data.meshes,
			bpy.data.lamps,
			bpy.data.cameras,
			):
		for id_data in bpy_data_iter:
			bpy_data_iter.remove(id_data)
			
def setup_scene():
	reset_scene()
	scene = context.scene
	
	camera_data = bpy.data.cameras.new("Camera")
	camera_data.type = 'PERSP'
	camera_data.clip_start = 0.01
	camera_data.clip_end = 10
	camera_data.lens = 35
	camera_data.lens_unit = 'MILLIMETERS'

	camA = bpy.data.objects.new("CamA", camera_data)
	camA.location = cam_pos_A
	camA.rotation_euler = cam_rot_A
	camA["render_size"] = texture_size
	scene.objects.link(camA)
	
	camB = bpy.data.objects.new("CamB", camera_data)
	camB.location = cam_pos_B
	camB.rotation_euler = cam_rot_B
	camB["render_size"] = texture_size
	scene.objects.link(camB)
	
	camO = bpy.data.objects.new("CamO", camera_data)
	camO["render_size"] = camera_size
	scene.objects.link(camO)
	
	light_data = bpy.data.lamps.new("Sun", type='SUN')
	
	light = bpy.data.objects.new("Sun", light_data)
	light.location = (0.0, 0.0, 1.0)
	scene.objects.link(light)
	
	light_settings = scene.world.light_settings
	light_settings.use_ambient_occlusion = True
	light_settings.ao_factor = 0.1
	light_settings.ao_blend_type = 'ADD'
	
	scene.update()
	
def render(camera, filepath, id=""):
	if id == "":
		id = camera.name
		
	context.scene.camera = camera								
	print("Rendering from %s" % (id))
	
	context.scene.render.filepath = "%s_%s.png" % (filepath, id)
	context.scene.render.resolution_x = camera["render_size"][0]
	context.scene.render.resolution_y = camera["render_size"][1]
	bpy.ops.render.render(write_still=True)
	
def render_all(filepath_src):
	filepath_render = os.path.splitext(filepath_src)[0]
	
	render(context.scene.objects["CamA"], filepath_render)
	render(context.scene.objects["CamB"], filepath_render)

	lerp_camera = context.scene.objects["CamO"]
	for lerp in np.arange(0, 1 + lerp_interval, lerp_interval):
		lerp_camera.location = cam_pos_A.lerp(cam_pos_B, lerp)
		lerp_camera.rotation_euler = cam_rot_A.lerp(cam_rot_B, lerp)
		render(lerp_camera, filepath_render, lerp)

def capture(files):
	for filepath_src in files:
		filepath_src = os.path.join(filepath_src, "models", "model_normalized.obj")
		filepath_dst = os.path.splitext(filepath_src)[0] + ".blend"
		setup_scene()

		print("Importing %s" % (filepath_src))
		bpy.ops.import_scene.obj(filepath=filepath_src, use_split_objects=False, use_split_groups=False)
		#bpy.ops.wm.save_as_mainfile(filepath=filepath_dst, check_existing=False)
		
		render_all(filepath_src)

if __name__ == "__main__":
	argv = sys.argv
	argv = argv[argv.index("--") + 1:]  # get all args after "--"
	capture(argv)