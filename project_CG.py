#Pakapol Sanarge 5810405223
import sys
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import pandas as pd
import math as m
import time
from PIL import Image
from matrixForTrans import *
import ctypes

win_w, win_h = 1024, 768
model_filenames = ["models/bunny.tri", "models/horse.tri", "models/objects_and_walls.tri" ]
models = []
params = {}
mouses = {}

shininess = 50
Ks = (1, 0.8, 0.5)

phong_id, groud_id,angle = 0, 0, 0
change_id = 0
main_id = phong_id

vao_render = 0

def print_shader_info_log(shader, prompt=""):
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not result:
        print("%s: %s" % (prompt, glGetShaderInfoLog(shader).decode("utf-8")))
        return -1
    else:
        return 0

def print_program_info_log(program, prompt=""):
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not result:
        print("%s: %s" % (prompt, glGetProgramInfoLog(program).decode("utf-8")))
        return -1
    else:
        return 0

def compile_program(vertex_code, fragment_code,shad):
    vert_id = glCreateShader(GL_VERTEX_SHADER)
    frag_id = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(vert_id, vertex_code)
    glShaderSource(frag_id, fragment_code)
    glCompileShader(vert_id)
    glCompileShader(frag_id)
    print_shader_info_log(vert_id, "Vertex Shader "+shad)
    print_shader_info_log(frag_id, "Fragment Shader "+shad)


    prog_id = glCreateProgram()
    glAttachShader(prog_id, vert_id)
    glAttachShader(prog_id, frag_id)

    glLinkProgram(prog_id)
    print_program_info_log(prog_id, "Link error")
    
    return prog_id  

def reshape(w, h):
    global win_w, win_h

    win_w, win_h = w, h
    glViewport(0, 0, w, h)  

wireframe, pause = False, True
def keyboard(key, x, y):
    global wireframe, pause , change_id,shininess,Ks

    key = key.decode("utf-8")
    if key == ' ':
        pause = not pause
        glutIdleFunc(None if pause else idle)

    elif key in ('p','g'): #Change shader Phong and Gouraud
        change_id = 0
        if key == 'g' :
            change_id = 1 

    elif key in ('n','N'): #Rise up and down Shininess
        rise = 1
        if key == "n":
            rise = -1
        shininess += rise

    elif key in ('s','S') : #Open and close specular
        Ks = (0,0,0)
        if key == "S":
            Ks = (1, 0.8, 0.5)

    elif key == 'W':
        wireframe = not wireframe
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if wireframe else GL_FILL)
    elif key in('q','Q') :
        exit(0)
    glutPostRedisplay()

#Mouse Controller
def mouse(button, state, x, y):
   mouses["button"] = button
   mouses["state"]  = state
   mouses["old_x"]  = x
   mouses["old_y"]  = y

def motion(x, y):
    current_x  = x
    current_y  = y
    
    if mouses["button"] == GLUT_LEFT_BUTTON: #Left
        params["rot_y"] += current_x - mouses["old_x"]
        params["rot_x"] += current_y - mouses["old_y"]
        print("Left")  

    elif mouses["button"] == GLUT_MIDDLE_BUTTON: #Middle
        move_x = normalize((mouses["old_x"] - current_x)/20)
        move_y = normalize((current_y - mouses["old_y"])/20)
        params["eye_pos"] += (move_x, move_y, 0)        
        params["eye_at"]  += (move_x, move_y, 0)
        print("Mid")

    elif mouses["button"] == GLUT_RIGHT_BUTTON: #Right
        move_z = normalize((current_y - mouses["old_y"])/20)
        params["eye_pos"] += (0, 0, move_z)        
        params["eye_at"]  += (0, 0, move_z)  
        print("Right")             
    glutPostRedisplay()
    mouses["old_x"]  = x
    mouses["old_y"]  = y


tick1, tick2 = 0, 0
def idle():
    global tick1, tick2

    tick1 += 1
    tick2 += 5
    glutPostRedisplay()

def display():
    print("%.2f fps" % (tick1/(time.time()-start_time)), tick1, tick2, end='\r')      
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    if change_id == 0:
        main_id = phong_id
    else:
        main_id = groud_id


    eye_pos   = params["eye_pos"]
    eye_at    = params["eye_at"]
    vertices  = params["vertices"]
    normals   = params["normals"]
    colors    = params["colors"]
    texcoords = params["texcoords"]
    rot_y     = params["rot_y"]
    rot_z     = params["rot_z"]
    rot_x     = params["rot_x"]

    light_pos = (-5, 5, 20)
    rotate_x = Rotate(rot_x, 1, 0, 0)
    rotate_y = Rotate(rot_y, 0, 1, 0)
    rotate_z = Rotate(rot_z, 0, 0, 1)
    model_mat = rotate_x @ rotate_y @ rotate_z


    glUseProgram(main_id)


    view_mat = LookAt(*eye_pos, *eye_at, 0, 1, 0)
    view_mat_location = glGetUniformLocation(main_id, "view_mat")
    glUniformMatrix4fv(view_mat_location, 1, GL_TRUE, view_mat)

    model_mat_location = glGetUniformLocation(main_id, "model_mat")
    glUniformMatrix4fv(model_mat_location, 1, GL_TRUE, model_mat)

    proj_mat = Perspective(45, win_w/win_h, 0.01, 500)
    proj_mat_location = glGetUniformLocation(main_id, "proj_mat")
    glUniformMatrix4fv(proj_mat_location, 1, GL_TRUE, proj_mat)


    Ka = (0, 0, 0) #ambient
    I = (1, 1, 1) #light Intensity

    eye_pos_location = glGetUniformLocation(main_id, "eye_pos")
    glUniform3f(eye_pos_location, *eye_pos)
    light_pos_location = glGetUniformLocation(main_id, "light_pos")
    glUniform3f(light_pos_location, *light_pos)

    Ks_location = glGetUniformLocation(main_id, "Ks")
    glUniform3f(Ks_location, *Ks)
    Ka_location = glGetUniformLocation(main_id, "Ka")
    glUniform3f(Ka_location, *Ka)
    I_location = glGetUniformLocation(main_id, "I")
    glUniform3f(I_location, *I)
    shininess_location = glGetUniformLocation(main_id, "shininess")
    glUniform1f(shininess_location, shininess)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDrawBuffer(GL_BACK)
    glViewport(0, 0, win_w, win_h)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBindTexture(GL_TEXTURE_2D, change_id)

    glBindVertexArray(vao_render)
    glDrawArrays(GL_TRIANGLES, 0, len(vertices))
    glutSwapBuffers()

def create_vbo():
    global main_id, vao_render
    glUseProgram(main_id)
    vao_render = glGenVertexArrays(1)
    glBindVertexArray(vao_render)

    vbo = glGenBuffers(4)
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
    glBufferData(GL_ARRAY_BUFFER, params["vertices"], GL_STATIC_DRAW)
    location = glGetAttribLocation(main_id, "position")
    glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(location)

    glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
    glBufferData(GL_ARRAY_BUFFER, params["normals"], GL_STATIC_DRAW)
    location = glGetAttribLocation(main_id, "normal")
    if location != -1:
        glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(location)

    glBindBuffer(GL_ARRAY_BUFFER, vbo[2])
    glBufferData(GL_ARRAY_BUFFER, params["colors"], GL_STATIC_DRAW)
    location = glGetAttribLocation(main_id, "color")
    if location != -1:
        glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(location)

    glBindBuffer(GL_ARRAY_BUFFER, vbo[3])
    glBufferData(GL_ARRAY_BUFFER, params["texcoords"], GL_STATIC_DRAW)
    location = glGetAttribLocation(main_id, "texcoord")
    if location != -1:
        glVertexAttribPointer(location, 2, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(location)
        
def init():
    global phong_id , groud_id ,main_id
    vert_code = b'''
//Phong shading
#version 150
uniform mat4 model_mat, view_mat, proj_mat;
in vec3 position, color, normal;
out vec3 fPos, fCol, fNor;
void main()
{   
    gl_Position = proj_mat* view_mat* model_mat * vec4(position, 1);
    fPos = position;
    fCol = color;
    fNor = normal;
}
                '''
    frag_code = b'''  
#version 150
uniform vec3 eye_pos, light_pos, Ks, Ka, I;
uniform float shininess;
in vec3 fPos, fCol, fNor;
out vec4 gl_FragColor;
void main()
{   
    vec3 N = normalize(fNor);
    vec3 L = normalize(light_pos - fPos);
    vec3 Kd = fCol;
    vec3 diffuse = Kd*max(0,dot(L,N))*I ;

    vec3 V = normalize(eye_pos - fPos);
    vec3 R = -L + 2 * max(dot(L, N), 0) * N;
    vec3 specular = Ks * pow(max(dot(V, R), 0), shininess)*I;
    vec3 ambient = Ka*I;

    vec3 color = diffuse + specular + ambient;

    gl_FragColor = vec4(color,1);

}
    '''
  
    phong_id = compile_program(vert_code, frag_code,"Phong shader") 
    main_id = phong_id

    vert_code = b'''
//Gouraud shading
#version 150
uniform mat4 model_mat, view_mat, proj_mat;
uniform vec3 eye_pos, light_pos, Ks, Ka, I;
uniform float shininess;
in vec3 position, color, normal;
out vec3 fcolor;
void main()
{   
    gl_Position = proj_mat* view_mat* model_mat * vec4(position, 1);
    vec3 N = normalize(normal);
    vec3 L = normalize(light_pos - position);
    vec3 Kd = color;
    vec3 diffuse = Kd*max(0,dot(L,N))*I ;

    vec3 V = normalize(eye_pos - position);
    vec3 R = -L + 2 * max(dot(L, N), 0) * N;
    vec3 specular = Ks * pow(max(dot(V, R), 0), shininess)*I;
    vec3 ambient = Ka*I;
    fcolor = diffuse + specular + ambient;
}
                '''
    frag_code = b''' 
#version 150
in vec3 fcolor;
out vec4 gl_FragColor;
void main()
{   
    gl_FragColor = vec4(fcolor,1);

}
                ''' 

    groud_id = compile_program(vert_code, frag_code,"Gouraud shader")

def gl_init_models():
    global start_time

    glClearColor(0, 0, 0, 0)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)


    for i in range(len(model_filenames)):
        df = pd.read_table(model_filenames[i], delim_whitespace=True, comment='#', header=None)
        centroid = df.values[:, 0:3].mean(axis=0)
        bbox = df.values[:, 0:3].max(axis=0) - df.values[:, 0:3].min(axis=0)

        vertices = np.ones((len(df.values), 3), dtype=np.float32)
        normals = np.zeros((len(df.values), 3), dtype=np.float32)
        texcoords = np.zeros((len(df.values), 2), dtype=np.float32)

        if len(df.values[0]) == 8:
            vertices[:, 0:3] = df.values[:, 0:3]
            normals[:, 0:3] = df.values[:, 3:6]
            texcoords[:, 0:2] = df.values[:, 6:8]
            colors = 0.5*(df.values[:, 3:6].astype(np.float32) + 1)
        else :
            colors = np.zeros((len(df.values), 3), dtype=np.float32)
            vertices[:, 0:3] = df.values[:, 0:3]
            colors[:, 0:3] = df.values[:, 3:6]
            normals[:, 0:3] = df.values[:, 6:9]
            texcoords[:, 0:2] = df.values[:, 9:11]


        models.append({"vertices": vertices, "normals": normals, 
                       "colors": colors, "texcoords": texcoords,
                       "centroid": centroid, "bbox": bbox})
        print("Model: %s, no. of vertices: %d, no. of triangles: %d" % 
               (model_filenames[i], len(vertices), len(vertices)//3))
        print("Centroid:", centroid)
        print("BBox:", bbox)
    model_id = 2
    centroid = params["centroid"] = models[model_id]["centroid"]
    bbox     = params["bbox"]     = models[model_id]["bbox"]
    params["vertices"]  = models[model_id]["vertices" ]
    params["normals"]   = models[model_id]["normals" ]
    params["colors"]    = models[model_id]["colors"] 
    params["texcoords"] = models[model_id]["texcoords"]
    params["eye_pos"]   = np.array((centroid[0], centroid[1], centroid[2]+1.5*bbox[0]), dtype=np.float32)
    params["eye_at"]    = np.array((centroid[0], centroid[1], centroid[2]), dtype=np.float32)
    params["rot_y"]     = 0
    params["rot_z"]     = 0
    params["rot_x"]     = 0
    start_time = time.time() - 0.0001
    
def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(win_w, win_h)
    glutCreateWindow(b"GLSL Template")
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutIdleFunc(idle)
    gl_init_models()
    init()
    create_vbo()
    glutMainLoop()

if __name__ == "__main__":
    main()