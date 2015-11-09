#version 330 core

// Interpolated values from the vertex shaders
//in vec2 UV;
in vec3 Position_worldspace;
in vec3 Normal_cameraspace;
in vec3 EyeDirection_cameraspace;
in vec3 LightDirection_cameraspace;
in vec4 ShadowCoord;

// Ouput data
out vec3 color;

// Values that stay constant for the whole mesh.
//uniform sampler2D myTextureSampler;
uniform mat4 MV;
uniform vec3 LightPosition_worldspace;
uniform sampler2DShadow shadowMap;
//uniform sampler2D shadowMap;

vec2 poissonDisk[16] = vec2[]( 
   vec2( -0.94201624, -0.39906216 ), 
   vec2( 0.94558609, -0.76890725 ), 
   vec2( -0.094184101, -0.92938870 ), 
   vec2( 0.34495938, 0.29387760 ), 
   vec2( -0.91588581, 0.45771432 ), 
   vec2( -0.81544232, -0.87912464 ), 
   vec2( -0.38277543, 0.27676845 ), 
   vec2( 0.97484398, 0.75648379 ), 
   vec2( 0.44323325, -0.97511554 ), 
   vec2( 0.53742981, -0.47373420 ), 
   vec2( -0.26496911, -0.41893023 ), 
   vec2( 0.79197514, 0.19090188 ), 
   vec2( -0.24188840, 0.99706507 ), 
   vec2( -0.81409955, 0.91437590 ), 
   vec2( 0.19984126, 0.78641367 ), 
   vec2( 0.14383161, -0.14100790 ) 
);

void main(){

	// Light emission properties
	// You probably want to put them as uniforms
	vec3 LightColor = vec3(1,1,1);
	float LightPower = 100.0f;
	
	// Material properties
	//vec3 MaterialDiffuseColor = texture2D( myTextureSampler, UV ).rgb;
	//vec3 MaterialAmbientColor = vec3(0.1,0.1,0.1) * MaterialDiffuseColor;
	vec3 MaterialDiffuseColor = vec3(0.5f,0.5f,0.5f);
	vec3 MaterialAmbientColor = vec3(0.0,0.0,0.0) * MaterialDiffuseColor;
	vec3 MaterialSpecularColor = vec3(0.3,0.3,0.3);

	// Distance to the light
	float distance = length( LightPosition_worldspace - Position_worldspace );

	// Normal of the computed fragment, in camera space
	vec3 n = normalize( Normal_cameraspace );
	// Direction of the light (from the fragment to the light)
	vec3 l = normalize( LightDirection_cameraspace );
	// Cosine of the angle between the normal and the light direction, 
	// clamped above 0
	//  - light is at the vertical of the triangle -> 1
	//  - light is perpendicular to the triangle -> 0
	//  - light is behind the triangle -> 0
	float cosTheta = clamp( dot( n,l ), 0,1 );
	
	// Eye vector (towards the camera)
	vec3 E = normalize(EyeDirection_cameraspace);
	// Direction in which the triangle reflects the light
	vec3 R = reflect(-l,n);
	// Cosine of the angle between the Eye vector and the Reflect vector,
	// clamped to 0
	//  - Looking into the reflection -> 1
	//  - Looking elsewhere -> < 1
	float cosAlpha = clamp( dot( E,R ), 0,1 );
	
	float visibility=1.0;

	// Fixed bias, or...
	float bias = 0.05;
	vec4 biasShadowCoord;
	//biasShadowCoord = vec4(ShadowCoord.xy, ShadowCoord.z-bias, ShadowCoord.w);
	//visibility = texture(shadowMap, vec3(ShadowCoord.xy, (ShadowCoord.z-bias)/ShadowCoord.w));
	//visibility = textureProj( shadowMap, biasShadowCoord);
	/*visibility = texture( 
					shadowMap, 
					vec3(ShadowCoord.xy,  
					(ShadowCoord.z-bias)/ShadowCoord.w)
					);*/
		

	int sampling_size = 10;	
	float portion = 1.0/sampling_size;
	for (int i=0; i < sampling_size; i++){
		// use either :
		//  - Always the same samples.
		//    Gives a fixed pattern in the shadow, but no noise
		int index = i;
		//  - A random sample, based on the pixel's screen location. 
		//    No banding, but the shadow moves with the camera, which looks weird.
		// int index = int(16.0*random(gl_FragCoord.xyy, i))%16;
		//  - A random sample, based on the pixel's position in world space.
		//    The position is rounded to the millimeter to avoid too much aliasing
		// int index = int(16.0*random(floor(Position_worldspace.xyz*1000.0), i))%16;
		
		// being fully in the shadow will eat up 4*0.2 = 0.8
		// 0.2 potentially remain, which is quite dark.
		biasShadowCoord = vec4(
						ShadowCoord.xy + poissonDisk[index]/70.0, 
						ShadowCoord.z-bias, 
						ShadowCoord.w);
		//visibility -= (1.0/sampling_size)*(1.0-texture( shadowMap, vec3(ShadowCoord.xy + poissonDisk[index]/700.0,  (ShadowCoord.z-bias)/ShadowCoord.w) ));
		visibility -= portion*(1.0-textureProj( shadowMap, biasShadowCoord));

	}
	/*
	vec4 shadowCoordinateWdivide = ShadowCoord / ShadowCoord.w ;
	
	float distanceFromLight = texture2D(shadowMap, shadowCoordinateWdivide.st).z;
	if (ShadowCoord.w > 0.0)
	 	visibility = distanceFromLight < shadowCoordinateWdivide.z ? 0.5 : 1.0 ;
	*/
		
	//if ( texture( shadowMap, (ShadowCoord.xy/ShadowCoord.w) ).z  <  (ShadowCoord.z-bias)/ShadowCoord.w )
	/*if( textureProj( shadowMap, ShadowCoord.xyw ).z < (ShadowCoord.z-bias)/ShadowCoord.w ){
		visibility = 0.0;	
	}*/
	
	/*
	if(visibility < 1.0)
		color = vec3(1.0 , 0.0 , 0.0);
	else*/
	//visibility = texture( shadowMap, vec3(ShadowCoord.xy, 0.0) );
	/*
	if(visibility < 1.0f) color = vec3(1.0, 0.0, 0.0);
	else if(visibility > 1.0f) color = vec3(1.0, 0.0, 1.0);
	else color = vec3(0.0, 0.0, 1.0);*/
	
	//color = vec3(0.0f, 0.0f, 1.0f);
	/*color = //MaterialAmbientColor +
		MaterialDiffuseColor * LightColor * LightPower * cosTheta / (distance*distance);
	if(visibility == 1.0){
		color = vec3(0,0,1);
	}*/
	
	//color = vec3(0,0,1);
	//color = visibility * MaterialDiffuseColor * LightColor;
	//visibility = 0.0f;
	/*color = 
		// Ambient : simulates indirect lighting
		MaterialAmbientColor +
		// Diffuse : "color" of the object
		visibility * MaterialDiffuseColor * LightColor * LightPower * cosTheta / (distance*distance) +
		// Specular : reflective highlight, like a mirror
		visibility * MaterialSpecularColor * LightColor * LightPower * pow(cosAlpha,5) / (distance*distance);
*/
	if(visibility < 1.0f) color = MaterialDiffuseColor*0.0f;
	else color = MaterialDiffuseColor*2.0f;
}