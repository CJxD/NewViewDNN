#version 130
#define MAX_LIGHTS 8

uniform mat4 CameraMVP;
uniform vec3 CameraPos;

uniform struct Material {
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
  
  uint mode;
  
  bool hasAmbientTex;
  bool hasDiffuseTex;
  bool hasSpecularTex;
  bool hasBumpTex;
} material;

uniform struct Light {
  vec3 position;
  float intensity;
} light[MAX_LIGHTS];
uniform uint numLights;

uniform sampler2D AmbientTex;
uniform sampler2D DiffuseTex;
uniform sampler2D SpecularTex;
uniform sampler2D BumpTex;

in vec3 ex_Position;
in vec3 ex_Normal;
in vec2 ex_UV;
out vec4 FragColor;

void main (void) 
{
  vec3 ambient, diffuse, specular;

  if (material.mode == 0u) {
    // Diffuse on, ambient and specular off (shading off)
    ambient = vec3(0, 0, 0);
	diffuse = material.hasDiffuseTex ? texture(DiffuseTex, ex_UV).rgb : material.diffuse;
	specular = vec3(0, 0, 0);
  }
  else {
    // Get normal offsets (converting from [0, 1] to [-1, 1]) if there's a bump map
    vec3 normalOffset = material.hasBumpTex ? (2 * texture(BumpTex, ex_UV).rgb) - 1 : vec3(0, 0, 0);
    
	vec3 normal = normalize(ex_Normal + normalOffset);
	vec3 viewDir = normalize(CameraPos - ex_Position);
    vec3 lightDir;
	
    float lambertian = 0.0;
    float specularity = 0.0;
 
    for (uint i = 0u; i < numLights; i++) {
      lightDir = normalize(light[i].position - ex_Position);

      float l = max(dot(lightDir, normal), 0.0);
      lambertian += l * light[i].intensity;
	  
      if (l > 0.0 && material.mode == 2u) {
        // Specular on (Blinn-Phong)
        vec3 halfDir = normalize(lightDir + viewDir);
        float specAngle = max(dot(halfDir, normal), 0.0);
        specularity += pow(specAngle, 16.0);
      }
    }

    ambient = material.hasAmbientTex ? texture(AmbientTex, ex_UV).rgb : material.ambient;
    diffuse = lambertian * (material.hasDiffuseTex ? texture(DiffuseTex, ex_UV).rgb : material.diffuse);
    specular = specularity * (material.hasSpecularTex ? texture(SpecularTex, ex_UV).rgb : material.specular);
  }
  
  FragColor = vec4(ambient + diffuse + specular, 1.0);
}