import bpy, bmesh
import os
import sys
import numpy as np

from glob import glob
from mathutils import Vector, Matrix

context = bpy.context

#
# Camera settings
#
camera_size = Vector((1024, 1024))
texture_size = Vector((1024, 1024))
num_captures = 6

mid_cam_pos = Vector((-1, 0, 0.1))
centre = Vector((0, 0, 0))
max_angle = np.pi / 3

#
# Mesh imperfection settings
#
mesh_simplify = 0.3
mesh_confidence = 0.9

def look_at(pos, target):
	'''Utility function returning a rotation matrix.
	Rotates an object at pos to look towards target.
	'''
	direction = target - pos
	rot_quat = direction.to_track_quat('-Z', 'Y')
	
	return rot_quat.to_matrix().to_4x4()
	
def yaw_look_at(pos, target, yaw):
	'''Utility function returning a full transformation matrix.
	Translates an object to pos, rotates to look towards target,
	then applies a yaw rotation. A negative value of
	yaw rotates left, and positive rotates right.
	'''
	local_rot = look_at(pos, target)
	trans = Matrix.Translation(pos)
	yaw = Matrix.Rotation(yaw, 4, Vector((0,0,1)))
	
	return yaw * trans * local_rot

def reset_scene():
	'''Utility function to remove all objects from the current scene.'''
	bpy.ops.wm.read_factory_settings()
	bpy.ops.wm.addon_enable(module="uv_perspective_project")

	for scene in bpy.data.scenes:
		for obj in scene.objects:
			scene.objects.unlink(obj)

	for bpy_data_iter in (
			bpy.data.objects,
			bpy.data.meshes,
			bpy.data.lamps,
			bpy.data.cameras,
			bpy.data.materials,
			bpy.data.images
			):
		for id_data in bpy_data_iter:
			bpy_data_iter.remove(id_data)
			
def setup_scene():
	'''"Lights, camera, ..."'''
	scene = context.scene
	
	# Lighting
	scene.world.horizon_color = (1.0, 1.0, 1.0)	
	light_settings = scene.world.light_settings
	light_settings.use_environment_light = True
	light_settings.environment_energy = 1
	light_settings.use_ambient_occlusion = True
	light_settings.ao_factor = 0.1
	light_settings.ao_blend_type = 'ADD'
	
	# Cameras
	camera_data = bpy.data.cameras.new("Camera")
	camera_data.type = 'PERSP'
	camera_data.clip_start = 0.01
	camera_data.clip_end = 10
	camera_data.lens = 35
	camera_data.lens_unit = 'MILLIMETERS'

	# Projection cameras
	cam0 = bpy.data.objects.new("Cam0", camera_data)
	cam0.matrix_world = yaw_look_at(mid_cam_pos, centre, -max_angle)
	cam0["render_size"] = texture_size
	scene.objects.link(cam0)
	cam1 = bpy.data.objects.new("Cam1", camera_data)
	cam1.matrix_world = yaw_look_at(mid_cam_pos, centre, max_angle)
	cam1["render_size"] = texture_size
	scene.objects.link(cam1)
	
	# Moveable intermediate camera
	camC = bpy.data.objects.new("CamC", camera_data)
	camC["render_size"] = camera_size
	scene.objects.link(camC)
	
	scene.update()
	
def preprocess(object):
	'''Fixes object inconsistencies.'''
	context.scene.objects.active = object
	bpy.ops.object.mode_set(mode='EDIT')
	
	# Make surface normals consistent
	bpy.ops.mesh.select_all(action='SELECT')
	bpy.ops.mesh.normals_make_consistent(inside=False)
	
	bpy.ops.object.mode_set(mode='OBJECT')
	
def imperfect(object, origins = [Vector((0,0,0))]):
	'''Warps an object to emulate imperfections in depth estimation.
	origins is a list of camera locations such that depth uncertainty
	is added along the vector between object vertex and camera location.
	'''
	context.scene.objects.active = object
	
	# Apply decimate modifier
	mod = object.modifiers.new("Simplify", 'DECIMATE')
	mod.decimate_type = 'DISSOLVE'
	mod.angle_limit = np.pi * mesh_simplify
	bpy.ops.object.modifier_apply(apply_as='DATA', modifier=mod.name)
	
	bpy.ops.object.mode_set(mode='EDIT')
	me = object.data
	bm = bmesh.from_edit_mesh(me)
	
	# Deform vertices along camera orientations
	q = (1 - mesh_confidence) / 50
	if q > 0:
		for v in bm.verts:
			# Warp the mesh along one choice of vector
			# by a gaussian with q factor derived from 1 - confidence
			vector = origins[np.random.choice(2)] - v.co
			vector.normalize()
			v.co += vector * np.random.normal(0, q)
		
	bmesh.update_edit_mesh(me)
	bpy.ops.object.mode_set(mode='OBJECT')

def project(images, object):
	'''Re-texturizes an object with projected textures
	Requires the separate UV Perspective Project addon
	'''
	scene = context.scene
	object.data.materials.clear()

	uvp = scene.uvPerspectiveProject
	uvp.cameras_settings.clear()
	uvp.object_name = object.name
	
	for i, image in enumerate(images):
		# UV setup
		uv = object.data.uv_textures.new("Proj" + str(i))
	
		# Texture setup
		tex = bpy.data.textures.new("Tex" + str(i), type='IMAGE')
		tex.image = image
	
		# Material setup
		mat = bpy.data.materials.new("Mat" + str(i))
		mat.use_shadeless = True
		mtex = mat.texture_slots.add()
		mtex.texture = tex
		mtex.texture_coords = 'UV'
		mtex.uv_layer = "Proj" + str(i)
		object.data.materials.append(mat)

		# UV Perspective Project addon
		uvp.cameras_settings.add()
		uvp.cameras_settings[-1].camera_name = "Cam" + str(i)
		uvp.cameras_settings[-1].uv_map_name = uv.name
		uvp.cameras_settings[-1].material_slot_name = mat.name
	
	bpy.ops.object.uvperspectiveprojectoperator('EXEC_DEFAULT')

def render(camera, filepath, id=""):
	'''Renders the scene from a camera.'''
	if id == "":
		id = camera.name
		
	context.scene.camera = camera								
	print("Rendering from %s" % (id))
	
	context.scene.render.image_settings.file_format = 'PNG'
	context.scene.render.image_settings.color_mode = 'RGB'
	
	context.scene.render.filepath = "%s_%s.png" % (filepath, id)
	context.scene.render.resolution_x = camera["render_size"][0]
	context.scene.render.resolution_y = camera["render_size"][1]
	context.scene.render.resolution_percentage = 100
	bpy.ops.render.render(write_still=True)
	
def render_lerp(filepath):
	'''Renders all camera locations between Cam0 and Cam1'''
	lerp_camera = context.scene.objects["CamC"]
	for i, lerp in enumerate(np.linspace(-max_angle, max_angle, num=num_captures)):
		lerp_camera.matrix_world = yaw_look_at(mid_cam_pos, centre, lerp)
		render(lerp_camera, filepath, i / (num_captures - 1))

def capture(model):
	reset_scene()
		
	# Find first .obj file
	obj_files = [y for x in os.walk(model) for y in glob(os.path.join(x[0], '*.obj'))]
	if len(obj_files) == 0:
		raise IOError("No object files found in " + model)
	
	filepath_src = obj_files[0]
	filepath_dst = os.path.join(model, "renders")
	if not os.path.exists(filepath_dst):
		os.makedirs(filepath_dst)
		
	# Paths for pre-projection and post-projection
	filepath_pre = os.path.join(filepath_dst, "view")
	filepath_post = os.path.join(filepath_dst, "proj")

	print("Importing %s" % (filepath_src))
	bpy.ops.import_scene.obj(filepath=filepath_src, use_split_objects=False, use_split_groups=False)
	object = context.scene.objects[-1]
	
	setup_scene()
	cam0 = context.scene.objects["Cam0"]
	cam1 = context.scene.objects["Cam1"]
	
	# Pre-process object inconsistencies
	preprocess(object)
	
	# Pre-projection
	render(cam0, filepath_pre)
	render(cam1, filepath_pre)
	render_lerp(filepath_pre)
	
	# Projection
	images = []
	images.append(bpy.data.images.load(filepath_pre + "_Cam0.png"))
	images.append(bpy.data.images.load(filepath_pre + "_Cam1.png"))
	imperfect(object, [cam0.location, cam1.location])
	project(images, object)
	
	# Post-projection
	render_lerp(filepath_post)
	
	# Optional: save .blend file
	#bpy.ops.wm.save_as_mainfile(filepath=filepath_dst + ".blend", check_existing=False)
	
def main(args):
	with open('view_interpolate.log', 'w') as log:
		for model in args:
			capture(model)
			print(model, file=log)

if __name__ == "__main__":
	argv = sys.argv
	argv = argv[argv.index("--") + 1:]  # get all args after "--"
	main(argv)