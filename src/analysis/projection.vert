#version 130

in vec3 VertexPosition;

out vec4 Proj1TexCoord;
out vec4 Proj2TexCoord;

uniform mat4 Projector1MVP;
uniform mat4 Projector2MVP;
uniform mat4 CameraMVP;

void main()
{
	vec4 pos4 = vec4(VertexPosition, 1.0);
	
	Proj1TexCoord = Projector1MVP * pos4;
	Proj2TexCoord = Projector2MVP * pos4;

	gl_Position = CameraMVP * pos4;
}