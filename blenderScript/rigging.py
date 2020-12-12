import sys

#insert path to your dope repository
sys.path.append("D:/Pierre/Documents/Polytechnique/Informatique/INF573/dope")
import webcam
import geometry
import bpy
import time
import numpy


class Rigging(bpy.types.Operator):
	bl_idname = "wm.rigging"
	bl_label = "DOPE Animation Operator"
	_timer = None

	# The main "loop"
	def modal(self, context, event):

		if (event.type in {'RIGHTMOUSE', 'ESC'}):
			bpy.context.active_object.animation_data_clear()
			self.cancel(context)
			return {'CANCELLED'}

		if event.type == 'TIMER':  # loop
			self.DOPE();

			bones = bpy.data.objects["Armature"].pose.bones
			print(format(bpy.context.scene.frame_current) + " / " + format(bpy.context.scene.frame_end))
			# print( bones["Base HumanPelvis"].rotation_axis_angle.data)
			bones["Base HumanPelvis"].rotation_mode = "XYZ";
			bones["Base HumanPelvis"].rotation_euler[0] += 0.001
			bones["Base HumanPelvis"].rotation_euler[1] += 0.001
			bones["Base HumanPelvis"].rotation_euler[2] += 0.001
			bones["Base HumanPelvis"].keyframe_insert(data_path="rotation_euler", index=-1)

		return {'PASS_THROUGH'}

	def DOPE(self):
		print("test")

	def stop_playback(self, scene):
		print(format(scene.frame_current) + " / " + format(scene.frame_end))
		if scene.frame_current == scene.frame_end:
			bpy.ops.screen.animation_cancel(restore_frame=False)

	def execute(self, context):
		bpy.app.handlers.frame_change_pre.append(self.stop_playback)
		wm = context.window_manager
		self._timer = wm.event_timer_add(0.01, window=context.window)
		wm.modal_handler_add(self)
		return {'RUNNING_MODAL'}

	def cancel(self, context):
		wm = context.window_manager
		wm.event_timer_remove(self._timer)


def register():
	bpy.utils.register_class(Rigging)


def unregister():
	bpy.utils.unregister_class(Rigging)


if __name__ == "__main__":
	register()

# test call
# bpy.ops.riggingAlgo()

