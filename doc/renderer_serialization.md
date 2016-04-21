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

## External Dependencies
- Serialization Format
	- option: yaml, json, pickle
- Renderer Dispatcher handling
- Components decoupling
 	- saving the optimization result, scene itself is sufficient
 	- saving the optimization setting, parameters is needed
 	- each components should be capable of saving/loading itself independently
- A external list of combos

## Refactoring process
1. Decoupling current codes;
2. Design and implement the save/load interface for each component and renderer as total;
3. Design and implement API;
4. create combo samples;