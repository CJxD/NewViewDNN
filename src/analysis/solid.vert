#version 130

uniform mat4 CameraMVP;

in vec3 in_Position;
in vec3 in_Normal;
in vec2 in_UV;

out vec3 ex_Position;
out vec3 ex_Normal;
out vec2 ex_UV;

void main(void)
{
    gl_Position = CameraMVP * vec4(in_Position, 1.0);
	ex_Position = in_Position;
	ex_Normal = in_Normal;
	ex_UV = in_UV;
}
