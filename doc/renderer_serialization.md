# Renderer Serialization

## Components
- Scene
	- casters
	- receivers
	- light(s)
 camera
    - camera for observation
    - camera for capturing
- parameter setting
    - dimension (should be implicit or else potentially conflicting)
    - getter/setter closure selection
    - parameter mask
- target shadow
- Windows layout
	- scene viewport size and position
	- capturing viewport size and position (which dictate the target image size)
	- control/spectating panel
- Outer dependencies
	- shader files
	- object files

## Outer Dependencies
- Serialization Format
	- option: yaml, json, pickle
- Renderer Dispatcher handling
- Components decoupling
 	- saving the optimization result, scene itself is sufficient
 	- saving the optimization setting, parameters is needed
 	- 