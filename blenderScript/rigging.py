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
			self.cancel(context)
			return {'CANCELLED'}

		if event.type == 'TIMER':  # loop
			self.init_DOPE();


		return {'PASS_THROUGH'}

	def init_DOPE(self):
		print("test")

	def stop_playback(self, scene):
		print(format(scene.frame_current) + " / " + format(scene.frame_end))
		if scene.frame_current == scene.frame_end:
			# Cancel animation, returning to the original frame
			bpy.ops.screen.animation_cancel(restore_frame=False)

	def execute(self, context):
		# This makes it possible to change data and relations (for example swap an object to another mesh) for the new
		# frame. Note that this handler is not to be used as ‘before the frame changes’ event.
		bpy.app.handlers.frame_change_pre.append(self.stop_playback)
		wm = context.window_manager
		# Add a timer to the given window, to generate periodic ‘TIMER’ events
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
    bpy.ops.riggingAlgo()