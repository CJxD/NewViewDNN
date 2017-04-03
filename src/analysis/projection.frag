#version 130

in vec4 Proj1TexCoord;
in vec4 Proj2TexCoord;

uniform sampler2D Proj1Tex;
uniform sampler2D Proj2Tex;

out vec4 FragColor;

void main() {
	vec4 colour = vec4(0.0);
	if (Proj1TexCoord.z > 0.0)
		colour = textureProj(Proj1Tex, Proj1TexCoord);
	//if (Proj2TexCoord.z > 0.0)
	//	colour = textureProj(Proj2Tex, Proj2TexCoord);
		
	FragColor = colour;
}