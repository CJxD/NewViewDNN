#version 130

in vec4 ex_Proj1TexCoord;
in vec4 ex_Proj2TexCoord;

uniform sampler2D Proj1Tex;
uniform sampler2D Proj2Tex;

out vec4 FragColor;

void main() {
	vec4 colour = vec4(0.0);
	if (ex_Proj1TexCoord.z > 0.0)
		colour = textureProj(Proj1Tex, ex_Proj1TexCoord);
	//if (ex_Proj2TexCoord.z > 0.0)
	//	colour = textureProj(Proj2Tex, ex_Proj2TexCoord);
		
	FragColor = colour;
}