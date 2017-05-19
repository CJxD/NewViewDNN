#version 130

in vec3 in_Position;

out vec4 ex_Proj1TexCoord;
out vec4 ex_Proj2TexCoord;

uniform mat4 Projector1MVP;
uniform mat4 Projector2MVP;
uniform mat4 CameraMVP;

void main()
{
	vec4 pos4 = vec4(in_Position, 1.0);
	
	ex_Proj1TexCoord = Projector1MVP * pos4;
	ex_Proj2TexCoord = Projector2MVP * pos4;

	gl_Position = CameraMVP * pos4;
}