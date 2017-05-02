bl_info = {
    "name": "UV Perspective Project",
    "author": "Lemon",
    "version": (0, 1),
    "blender": (2, 72, 0),
    "location": "View3D > Object",
    "description": "Multi-image perspective projecting onto an object",
    "category": "Object"}
	
import bpy
import time
from mathutils import Vector
from operator import itemgetter

#Used to store information related to each camera setting
class UVPPCameraSetting:
    def __init__( self, scene, camera, obj, uvMap, materialSlotIndex ):
        self.camera = camera
        self.uvMap = uvMap
        self.materialSlotIndex = materialSlotIndex
        #Z axis of the camera translate in world (Z axis is opposite to view)
        self.zInWorld = Vector( (0, 0, 1) )
        self.zInWorld.rotate( self.camera.matrix_world.to_euler() )
        #To keep polygons to handle for this setting
        self.polygons = set()
        #Camera parameters
        #Matrix to convert from object coordinate to camera coordinates
        self.toCameraMatrix = camera.matrix_world.inverted() * obj.matrix_world
        #The frame is composed of the coordinates in the camera view
        frame = [v / v.z for v in camera.data.view_frame(scene=scene)]
        #Get the X, Y corners
        self.minX = min( v.x for v in frame )
        self.maxX = max( v.x for v in frame )
        self.minY = min( v.y for v in frame )
        self.maxY = max( v.y for v in frame )
        #Precalculations to avoid to repeat them when applied to the model
        self.deltaX = self.maxX - self.minX
        self.deltaY = self.maxY - self.minY
        self.offsetX = self.minX / self.deltaX
        self.offsetY = self.minY / self.deltaY

    #Calculate the UV coordinateds from the object coordinates
    def CalcUV( self, objCo ):
        #Object in camera view
        camCo = self.toCameraMatrix * objCo
        #Z is "inverted" as camera view is pointing to -Z of the camera
        z = -camCo.z
        try:
            #Translates x and y to UV coordinates
            x = (camCo.x / (self.deltaX * z)) - self.offsetX        
            y = (camCo.y / (self.deltaY * z)) - self.offsetY        
            return x, y, z
        except:
            #In case Z is zero
            return 0.5, 0.5, 0

def UVPerspectiveProject( scene, obj, camSettingList ):
    print( '------------------------' )
    startTime = time.time()

    matrix_world = obj.matrix_world
    rotation_world = matrix_world.to_euler()

    #Assign polygon to its corresponding camera considering faces most aligned to camera view
    for p in obj.data.polygons:
        normalInWorld = Vector( p.normal )
        normalInWorld.rotate( rotation_world )
        camSetting, maxDot = max( ((c, normalInWorld.dot( c.zInWorld)) for c in camSettingList), key=itemgetter(1) )
        camSetting.polygons.add( p.index )

    print( 'precalculations. elapse in seconds', time.time() - startTime )

    loops = obj.data.loops
    vertices = obj.data.vertices
    #For each setting, 
    for cs in camSettingList:
        camera = cs.camera
        uvMap = cs.uvMap
        polygons = cs.polygons
        projectedCo = {} #Storage to avoid multiple calculations of the same world_to_camera_view
        #Go through all polygons
        for p in obj.data.polygons:
            #If the polygon corresponds to the setting
            if p.index in polygons:
                #Assign the material index
                p.material_index = cs.materialSlotIndex
                #Calculate each vertex uv projection
                for i, vi in [(i, loops[i].vertex_index) for i in p.loop_indices]:
                    if vi not in projectedCo: #not already calculated for this cam
                        x, y, z = cs.CalcUV( vertices[vi].co )
                        projectedCo[vi] = (x,y)
                    uvMap.data[i].uv = projectedCo[vi]
            #If not, uv are set to (0,0)
            else:
                for i in p.loop_indices:
                    uvMap.data[i].uv = (0, 0)

    print( 'done. elapse in seconds', time.time() - startTime )

def TargetObjExists( context ):
    try:
        scn = context.scene
        obj = scn.objects[scn.uvPerspectiveProject.object_name]
        return obj.type == 'MESH'
    except:
        pass
    return False

def IsCamera( context, camera_name ):
    try:
        camera = context.scene.objects[camera_name]
        return camera.type == 'CAMERA'
    except:
        pass
    return False

def SettingsAreOK( context ):
    try:
        scn = context.scene
        settings = scn.uvPerspectiveProject
        obj = scn.objects[settings.object_name]
        assert obj.type == 'MESH'

        for item in settings.cameras_settings:
            camera = scn.objects[item.camera_name]
            assert camera.type == 'CAMERA'
            uv_map = obj.data.uv_layers[item.uv_map_name]
            materialSlot = obj.material_slots[item.material_slot_name]

        return True
    except:
        pass
    return False

class UVPerspectiveProjectOperator( bpy.types.Operator ):
    bl_idname = "lemon.uvperspectiveprojectoperator"
    bl_label = "UV Perspective Project"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(self, context):
        return context.mode == 'OBJECT' and SettingsAreOK( context )

    def invoke(self, context, event):
        scn = context.scene
        settings = scn.uvPerspectiveProject

        #try:
        obj = scn.objects[settings.object_name]
        cameraSettings = [UVPPCameraSetting( scn, scn.objects[item.camera_name], obj, obj.data.uv_layers[item.uv_map_name], obj.material_slots.find( item.material_slot_name ) ) for item in settings.cameras_settings]
        UVPerspectiveProject( scn, obj, cameraSettings )
        #except:
            #pass

        return { 'FINISHED' }

class UVPerspectiveProjectCamSettingsActions( bpy.types.Operator ):
    bl_idname = "lemon.uvperspectiveprojectpanelcamlistactions"
    bl_label = ""

    action = bpy.props.EnumProperty( items = ( ('UP', "Up", ""), ('DOWN', "Down", ""), ('REMOVE', "Remove", ""), ('ADD', "Add", ""), ) )

    def invoke(self, context, event):

        scn = context.scene
        settings = scn.uvPerspectiveProject
        index = settings.camera_setting_index

        if self.action == 'DOWN' and index < len(settings.cameras_settings) - 1:
            settings.cameras_settings.move( index, index + 1 )
            settings.camera_setting_index += 1

        elif self.action == 'UP' and index > 0:
            settings.cameras_settings.move( index, index - 1 )
            settings.camera_setting_index -= 1

        elif self.action == 'REMOVE':
            settings.cameras_settings.remove( settings.camera_setting_index )
            settings.camera_setting_index -= 1

        elif self.action == 'ADD':
            item = settings.cameras_settings.add()
            settings.camera_setting_index = len(settings.cameras_settings) - 1

        return {"FINISHED"}

class UVPerspectiveProjectCamSettingsPanel( bpy.types.UIList ):

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):

        scn = context.scene
        settings = scn.uvPerspectiveProject

        try:
            obj = scn.objects[settings.object_name]
        except:
            pass

        suffix = " (" + str( index + 1 ) + ")"

        row = layout.row()
        col = row.column( align = True )
        col.prop_search( item, "camera_name", scn, "objects", text="Camera" + suffix  )
        col.prop_search( item, "uv_map_name", obj.data, "uv_layers", text="UV map" + suffix )
        col.prop_search( item, "material_slot_name", obj, "material_slots", text="Material" + suffix )

    def invoke(self, context, event):
        pass   

class UVPerspectiveProjectPanel( bpy.types.Panel ):
    bl_idname = 'lemon.uvperspectiveprojectpanel'
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_label = "UV Perspective Project"

    @classmethod
    def poll(self, context):
        return context.mode == 'OBJECT'

    def draw(self, context):
        layout = self.layout
        scn = context.scene

        settings = scn.uvPerspectiveProject

        layout.row().prop_search( settings, "object_name", scn, "objects", text="Object" )

        if TargetObjExists( context ):
            layout.row().label( text="Cameras settings:" )

            row = layout.row()        
            row.template_list( "UVPerspectiveProjectCamSettingsPanel", "", settings, "cameras_settings", settings, "camera_setting_index", rows = 3 )

            col = row.column( align = True )
            col.operator( UVPerspectiveProjectCamSettingsActions.bl_idname, icon='ZOOMIN', text="" ).action = 'ADD'
            col.operator( UVPerspectiveProjectCamSettingsActions.bl_idname, icon='ZOOMOUT', text="" ).action = 'REMOVE'
            col.separator()
            col.operator( UVPerspectiveProjectCamSettingsActions.bl_idname, icon='TRIA_UP', text="" ).action = 'UP'
            col.operator( UVPerspectiveProjectCamSettingsActions.bl_idname, icon='TRIA_DOWN', text="" ).action = 'DOWN'

            layout.row().operator( UVPerspectiveProjectOperator.bl_idname, text="Apply" )


class UVPerspectiveProjectCamSettingsProps( bpy.types.PropertyGroup ):
    camera_name = bpy.props.StringProperty()
    uv_map_name = bpy.props.StringProperty()
    material_slot_name = bpy.props.StringProperty()

class UVPerspectiveProjectProps( bpy.types.PropertyGroup ):
    object_name = bpy.props.StringProperty()
    cameras_settings = bpy.props.CollectionProperty( type = UVPerspectiveProjectCamSettingsProps )
    camera_setting_index = bpy.props.IntProperty()

def register():
    bpy.utils.register_module(__name__)
    bpy.types.Scene.uvPerspectiveProject = bpy.props.PointerProperty( type = UVPerspectiveProjectProps )

def unregister():
    del bpy.types.Scene.uvPerspectiveProject
    bpy.utils.unregister_module(__name__)

if __name__ == '__main__':
  register()