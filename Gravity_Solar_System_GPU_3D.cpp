#include <GL/glew.h> // OpenGL function loader (required for modern OpenGL)
#include <GLFW/glfw3.h>
#include <glm/glm.hpp> // GLM math library (vectors/matrices like GLSL)
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>               // Debug output
#define _USE_MATH_DEFINES
#include <string>                 // For file paths and uniform names
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <fstream>                // File reading
#include <sstream>                // String stream to load shader files
#ifndef M_PI
#define M_PI 3.14159265358979323846
#define G 6.6743e-11
#define c 299792458.0
#endif
using namespace glm;
using Clock = std::chrono::high_resolution_clock;
const double G_SIM = 4*M_PI*M_PI;
const double m_s = 1.989e30;
const double year = 60.0*60.0*24.0*365.0;
const double m_e = 6.371e3; // Mass of earth
const double au = 1.496e11; // Astronomical units

// Include Shader pack here

const char *sphereShaderSource = R"glsl(
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

out vec3 fragColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec3 objectColor;

void main()
{
    fragColor = objectColor;

    gl_Position =
        projection *
        view *
        model *
        vec4(aPos, 1.0);
}
)glsl";

const char *gridShaderSource = R"glsl(
#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 fragColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec3 objectPositions[32];
uniform float objectMasses[32];
uniform float objectRadii[32];
uniform int objectCount;

const float G_SIM = 39.4784;
const float c = 299792458.0;

float curvature(vec3 p)
{
    float totalY = 0.0;

    for(int i = 0; i < objectCount; i++)
    {
        vec3 obj = objectPositions[i];

        float dx = p.x - obj.x;
        float dz = p.z - obj.z;

        float r = sqrt(dx * dx + dz * dz);

        r = max(r, objectRadii[i]);

        float rs = (2.0 * G_SIM * objectMasses[i]) / (c * c);

        totalY += -5e7 * sqrt(rs / r);
    }

    return totalY;
}

void main()
{
    fragColor = aColor;

    vec3 pos = aPos;

    pos.y = curvature(pos);

    gl_Position =
        projection *
        view *
        model *
        vec4(pos, 1.0);
}
)glsl";

const char *fragmentShaderSource = R"glsl(
#version 330 core

in vec3 fragColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(fragColor, 1.0);
}
)glsl";


// This code has more accurate acceleration calculation that calculates ALL acting accelerations and adds them via superposition

struct Engine;
struct Vertex;
struct Celestial_Objects;
void gravityStep(std::vector<Celestial_Objects> &objects, double dlambda);
void gravity_newtonian(Celestial_Objects & obj, Celestial_Objects & other, double k[12]);
void gravityLeapfrog(std::vector<Celestial_Objects> &objects, const float dlambda);
void objectCollision(Celestial_Objects &obj, Celestial_Objects &other);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void focusCameraOnObject(Engine &engine, Celestial_Objects &obj);
void syncFreeCameraFromOrbit(Engine& engine);
void simulationReset(std::vector<Celestial_Objects> &objects, std::vector<Celestial_Objects> &objectsCopy);
glm::vec3 spherical2carteesian(float radius, float theta, float phi, double x, double y, double z);

// To control body position and color more easily
struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
};

class Shader {
public:
    GLuint ID; // This stores the compiled GPU shader program handle

    // Constructor: loads, compiles, and links shaders
    Shader(const char* vertexCode, const char* fragmentCode, bool fromSource) {

        // Convert C++ strings into C-style strings for OpenGL
        const char* vCode = vertexCode;
        const char* fCode = fragmentCode;

        GLuint vertex, fragment; // IDs for compiled shader objects

        // =========================
        // VERTEX SHADER COMPILATION
        // =========================

        vertex = glCreateShader(GL_VERTEX_SHADER); // Create shader object
        glShaderSource(vertex, 1, &vCode, nullptr); // Attach source code
        glCompileShader(vertex); // Compile shader on GPU
        checkCompileErrors(vertex, "VERTEX"); // Check for errors

        // ===========================
        // FRAGMENT SHADER COMPILATION
        // ===========================

        fragment = glCreateShader(GL_FRAGMENT_SHADER); // Create fragment shader
        glShaderSource(fragment, 1, &fCode, nullptr); // Attach source code
        glCompileShader(fragment); // Compile fragment shader
        checkCompileErrors(fragment, "FRAGMENT"); // Check errors

        // ======================
        // LINK SHADERS TOGETHER
        // ======================

        ID = glCreateProgram(); // Create shader program (final GPU pipeline object)

        glAttachShader(ID, vertex);   // Attach vertex shader
        glAttachShader(ID, fragment); // Attach fragment shader
        glLinkProgram(ID);            // Link them into a single program

        checkCompileErrors(ID, "PROGRAM"); // Check linking errors

        // Shaders are no longer needed after linking
        glDeleteShader(vertex);
        glDeleteShader(fragment);
    }

    // Activate this shader program
    void use() {
        glUseProgram(ID);
    }

    // Send a 4x4 matrix (used for model/view/projection)
    void setMat4(const std::string& name, const glm::mat4& mat) const {
        glUniformMatrix4fv(
            glGetUniformLocation(ID, name.c_str()), // Find uniform location in shader
            1,                                      // Number of matrices
            GL_FALSE,                               // No transpose (GLM already column-major)
            &mat[0][0]                              // Pointer to matrix data
        );
    }

    // Send a vec3 (used for color, position, etc.)
    void setVec3(const std::string& name, const glm::vec3& v) const {
        glUniform3fv(
            glGetUniformLocation(ID, name.c_str()), // Find uniform location
            1,                                      // One vector
            &v[0]                                   // Pointer to vector data
        );
    }

private:

    // Helper function to check shader compilation/linking errors
    void checkCompileErrors(GLuint shader, std::string type) {
        GLint success;         // Stores success/failure result
        GLchar infoLog[1024];  // Stores error message

        if (type != "PROGRAM") {
            // Check shader compile errors
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

            if (!success) {
                glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
                std::cerr << "Shader compile error (" << type << "):\n"
                          << infoLog << "\n";
            }
        } else {
            // Check program linking errors
            glGetProgramiv(shader, GL_LINK_STATUS, &success);

            if (!success) {
                glGetProgramInfoLog(shader, 1024, nullptr, infoLog);
                std::cerr << "Program linking error:\n"
                          << infoLog << "\n";
            }
        }
    }
};

struct Celestial_Objects {
    double x, y, z;
    double vx, vy, vz;
    double r, vr, ar;
    double phi, vphi, aphi;
    double r_render;
    double mass;
    double r_co;
    glm::vec3 color;

    // Celectial Object definition
    Celestial_Objects(vec3 pos, vec3 vel, double m, double r_co, double r_render, glm::vec3 color = glm::vec3(1.0f, 0.0f, 0.0f)) : x(pos.x), y(pos.y), z(pos.z), vx(vel.x), vy(vel.y), vz(vel.z), mass(m), r_co(r_co), r_render(r_render), color(color) {
        r = sqrt(x*x + y*y + z*z);
        phi = atan2(y, x);
        vr = vx*cos(phi) + vy * sin(phi);
        vphi = (-vx * sin(phi) + vy * cos(phi))/r;
        ar = 0.0;
        aphi = 0.0;
    }
};


struct Engine {
    GLFWwindow* window;
    int WIDTH = 3840; // Pixel Width, Pixel Height
    int HEIGHT = 2160;
    float width = 1e6; // Position width, position height
    float height = 5.625e5;

    // Camera variables
    // Position of camera in world space
    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);

    // Direction camera is facing
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);

    // Up direction of camera
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

    // Zoom factor
    float fov = 45.0f;

    // Euler angles for mouse rotation
    float yaw = -90.0f;
    float pitch = 0.0f;

    // Last mouse positions
    float lastX = WIDTH / 2.0f;
    float lastY = HEIGHT / 2.0f;

    // Prevent sudden jump on first mouse movement
    bool firstMouse = true;

    // Time between current and previous frame
    float deltaTime = 0.0f;

    // Previous frame timestamp
    float lastFrame = 0.0f;
    mat4 view;

    enum CameraMode {FREE, ORBIT};

    CameraMode cameraMode = FREE;
    int orbitTarget = -1; // index of celestial object
    float orbitDistance = 1.0f; // will be set per object
    float orbitYaw = 0.0f;
    float orbitPitch = 0.0f;

    // Defines the OpenGL engine that runs the window
    Engine() {
        if(!glfwInit()){
            std::cerr << "GLFW failed init!" << '\n';
            std::exit(EXIT_FAILURE);
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Gravity Simulation", nullptr, nullptr);

        if(!window){
            std::cerr << "Window could not be created!" << '\n';
            glfwTerminate();
            std::exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(window);

        glfwSetCursorPosCallback(window, mouse_callback);

        glfwSetScrollCallback(window, scroll_callback);

        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        glEnable(GL_DEPTH_TEST);

        glewExperimental = GL_TRUE;

        GLenum err = glewInit();

        if(err != GLEW_OK){
            std::cerr << "GLEW failed init: "
                    << glewGetErrorString(err)
                    << '\n';
            std::exit(EXIT_FAILURE);
        }

        glViewport(0, 0, WIDTH, HEIGHT);

        glEnable(GL_PROGRAM_POINT_SIZE);

        glPointSize(10.0f);

        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
    }

    void processInput(std::vector<Celestial_Objects> &objects, std::vector<Celestial_Objects> &objectsCopy) {
        float cameraSpeed = 1.0f * deltaTime;

        if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            cameraPos += cameraSpeed * cameraFront;

        if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            cameraPos -= cameraSpeed * cameraFront;

        if(glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraUp;

        if(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraUp;

        if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            cameraPos -= glm::normalize(
                glm::cross(cameraFront, cameraUp)
            ) * cameraSpeed;

        if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            cameraPos += glm::normalize(
                glm::cross(cameraFront, cameraUp)
            ) * cameraSpeed;

        if ((glfwGetKey(window, GLFW_KEY_BACKSPACE) == GLFW_PRESS)
            && cameraMode == ORBIT)
        {
            syncFreeCameraFromOrbit(*this);

            cameraMode = FREE;
        }

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            simulationReset(objects, objectsCopy);
        }

        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
            cameraMode = ORBIT;
            orbitTarget = 0;
        }

        if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
            cameraMode = ORBIT;
            orbitTarget = 1;
        }

        if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
            cameraMode = ORBIT;
            orbitTarget = 2;
        }

        if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
            cameraMode = ORBIT;
            orbitTarget = 3;
        }

        if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
            cameraMode = ORBIT;
            orbitTarget = 4;
        }

        if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS) {
            cameraMode = ORBIT;
            orbitTarget = 5;
        }

        if (glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS) {
            cameraMode = ORBIT;
            orbitTarget = 6;
        }

        if (glfwGetKey(window, GLFW_KEY_8) == GLFW_PRESS) {
            cameraMode = ORBIT;
            orbitTarget = 7;
        }

        if (glfwGetKey(window, GLFW_KEY_9) == GLFW_PRESS) {
            cameraMode = ORBIT;
            orbitTarget = 8;
        }

        if (glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS) {
            cameraMode = ORBIT;
            orbitTarget = 9;
        }
    }

    // Actually creates the window
    void run(Shader& shader, std::vector<Celestial_Objects> &objects, std::vector<Celestial_Objects> &objectsCopy) {
        float currentFrame = glfwGetTime();

        deltaTime = currentFrame - lastFrame;

        lastFrame = currentFrame;

        processInput(objects, objectsCopy);

        glm::mat4 model = glm::mat4(1.0f);

        if (cameraMode == FREE)
        {
            view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        }

        else if (cameraMode == ORBIT && orbitTarget >= 0)
        {
            auto &obj = objects[orbitTarget];

            glm::vec3 target(obj.x, obj.y, obj.z);

            orbitDistance = 0.01f + obj.r_render * 20.0f; // tweak scale

            glm::vec3 offset;
            offset.x = orbitDistance * cos(glm::radians(orbitYaw)) * cos(glm::radians(orbitPitch));
            offset.y = orbitDistance * sin(glm::radians(orbitPitch));
            offset.z = orbitDistance * sin(glm::radians(orbitYaw)) * cos(glm::radians(orbitPitch));

            cameraPos = target + offset;
            cameraFront = glm::normalize(target - cameraPos);

            view = glm::lookAt(cameraPos, target, cameraUp);
        }

        glm::mat4 projection = glm::perspective(
            glm::radians(fov),
            (float)WIDTH / (float)HEIGHT,
            1e-5f,
            1e9f
        );

        shader.setMat4("model", model);
        shader.setMat4("view", view);
        shader.setMat4("projection", projection);
    }

};


struct SimulationContext {
    Engine* engine;
    std::vector<Celestial_Objects>* objects;
};

// Needs to be its own function due to gravitational pull from multiple objects
void gravityStep(std::vector<Celestial_Objects> &objects, double dlambda) {

    // for(auto & obj : objects) obj.detectScreenCollision();

    gravityLeapfrog(objects, dlambda);

}
// Leap Frog solver


glm::vec3 computeAcceleration(const Celestial_Objects& obj,
                              const std::vector<Celestial_Objects>& objects)
{
    glm::vec3 acc(0.0f);

    for (const auto& other : objects)
    {
        if (&obj == &other) continue;

        glm::vec3 r(
            other.x - obj.x,
            other.y - obj.y,
            other.z - obj.z
        );

        double dist2 = glm::dot(r, r) + 1e-12; // softening
        double dist  = sqrt(dist2);

        double force = G_SIM * other.mass / (dist2 * dist);

        acc += (float)force * r;
    }

    return acc;
}

void gravityLeapfrog(std::vector<Celestial_Objects>& objects, float dt)
{
    size_t n = objects.size();

    // Compute initial accelerations
    std::vector<glm::vec3> acc(n);

    for (size_t i = 0; i < n; i++)
        acc[i] = computeAcceleration(objects[i], objects);

    // Half velocity kick
    for (size_t i = 0; i < n; i++)
    {
        objects[i].vx += 0.5 * dt * acc[i].x;
        objects[i].vy += 0.5 * dt * acc[i].y;
        objects[i].vz += 0.5 * dt * acc[i].z;
    }

    // Full position drift
    for (size_t i = 0; i < n; i++)
    {
        objects[i].x += dt * objects[i].vx;
        objects[i].y += dt * objects[i].vy;
        objects[i].z += dt * objects[i].vz;
    }

    // Recompute accelerations
    std::vector<glm::vec3> newAcc(n);

    for (size_t i = 0; i < n; i++)
        newAcc[i] = computeAcceleration(objects[i], objects);

    // Second half velocity kick
    for (size_t i = 0; i < n; i++)
    {
        objects[i].vx += 0.5 * dt * newAcc[i].x;
        objects[i].vy += 0.5 * dt * newAcc[i].y;
        objects[i].vz += 0.5 * dt * newAcc[i].z;
    }
}

// Collision detector

void objectCollision(Celestial_Objects &obj, Celestial_Objects &other) {

    // Define relative distance between objects
    double dx = other.x - obj.x;
    double dy = other.y - obj.y;
    double dz = other.z - obj.z;
    double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    // Distance checker between two objects
    double minDist = (obj.r_co + other.r_co);

    // Skip collicion if ojects arent touching or if they are too close
    if (dist >= minDist || dist < 1e-8) return;

    // Velocity normal vector for obj
    double nx = dx / dist;
    double ny = dy / dist;
    double nz = dz / dist;

    // Relative velocity vector
    double dvx = other.vx - obj.vx;
    double dvy = other.vy - obj.vy;
    double dvz = other.vz - obj.vz;

    // Dot product of relative velocity vector and normal/tangent vector of obj
    double relVel = dvx * nx + dvy * ny + dvz * nz;

    // Tangent vector = vrel - (vrel * n)n
    double tx = dvx - relVel * nx;
    double ty = dvy - relVel * ny;
    double tz = dvz - relVel * nz;
    double t_size = sqrt(tx*tx + ty*ty + tz*tz);

    // Normal directions for tangent
    double ntx = tx / t_size;
    double nty = ty / t_size;
    double ntz = tz / t_size;

    // If dot product is positive -> Objects are moving away -> dont apply collision
    if (relVel > 0) return;

    // Define masses
    double m1 = obj.mass;
    double m2 = other.mass;

    // Define impulse. Restitution is a damping factor that is material dependent
    double restitution = 0.9; // Controlls collision damping
    double impulse = (1 + restitution) * relVel / (1/m1 + 1/m2);

    // Change velocities of objects
    obj.vx += impulse / m1 * nx;
    obj.vy += impulse / m1 * ny;
    obj.vz += impulse / m1 * nz;

    other.vx -= impulse / m2 * nx;
    other.vy -= impulse / m2 * ny;
    other.vz -= impulse / m2 * nz;

    // Tangent friction
    double friction = 0.05; // tune this

    double tangentImpulse = friction * t_size / (1.0/m1 + 1.0/m2);

    obj.vx += tangentImpulse * ntx / m1;
    obj.vy += tangentImpulse * nty / m1;
    obj.vz += tangentImpulse * ntz / m1;

    other.vx -= tangentImpulse * ntx / m2;
    other.vy -= tangentImpulse * nty / m2;
    other.vz -= tangentImpulse * ntz / m2;

    // POSITION correction (separate, clean)
    double overlap = minDist - dist;
    double margin = 0.01;

    if (overlap > margin) {
        double correction = overlap - margin;
        double percent = 0.8;

        obj.x -= correction * percent * nx;
        obj.y -= correction * percent * ny;
        obj.z -= correction * percent * nz;

        other.x += correction * percent * nx;
        other.y += correction * percent * ny;
        other.z += correction * percent * nz;
    }
}

void runVAOVBO(std::vector<Vertex>& vertices, GLuint& VAO, GLuint& VBO) {

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(
        GL_ARRAY_BUFFER,
        vertices.size() * sizeof(Vertex),
        vertices.data(),
        GL_DYNAMIC_DRAW
    );

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, color));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

// Called automatically whenever mouse moves
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    // Retrieve engine pointer from GLFW window
    auto* ctx =
        static_cast<SimulationContext*>(glfwGetWindowUserPointer(window));

    Engine* engine = ctx->engine;

    // Prevent camera snapping on first frame
    if(engine->firstMouse)
    {
        engine->lastX = xpos;
        engine->lastY = ypos;
        engine->firstMouse = false;
    }

    // Calculate mouse movement offsets
    float xoffset = xpos - engine->lastX;
    float yoffset = engine->lastY - ypos;

    // Store current mouse position
    engine->lastX = xpos;
    engine->lastY = ypos;

    // Mouse sensitivity scaling
    float sensitivity = 0.1f;

    xoffset *= sensitivity;
    yoffset *= sensitivity;

    // Update camera rotation angles
    engine->yaw += xoffset;
    engine->pitch += yoffset;

    // Prevent camera flip at vertical extremes
    if(engine->pitch > 89.0f)
        engine->pitch = 89.0f;

    if(engine->pitch < -89.0f)
        engine->pitch = -89.0f;

    if (engine->cameraMode == Engine::ORBIT)
    {
        float sensitivity = 0.1f;

        engine->orbitYaw   += xoffset * sensitivity;
        engine->orbitPitch += yoffset * sensitivity;

        if (engine->orbitPitch > 89.0f) engine->orbitPitch = 89.0f;
        if (engine->orbitPitch < -89.0f) engine->orbitPitch = -89.0f;
        return;
    }

    // Recalculate camera direction vector
    glm::vec3 front;

    front.x = cos(glm::radians(engine->yaw))
            * cos(glm::radians(engine->pitch));

    front.y = sin(glm::radians(engine->pitch));

    front.z = sin(glm::radians(engine->yaw))
            * cos(glm::radians(engine->pitch));

    // Normalize keeps vector length = 1
    engine->cameraFront = glm::normalize(front);
}


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS)
        return;

    auto* ctx =
        static_cast<SimulationContext*>(glfwGetWindowUserPointer(window));

    Engine& engine = *ctx->engine;
    auto& objects = *ctx->objects;

    glm::vec3 camPos = engine.cameraPos;
    glm::vec3 camVel = engine.cameraFront * 10.0f; // or tweak if you want “throwing suns”

    objects.emplace_back(camPos, camVel, 1.0, 0.005, 0.05, vec3(1.0f, 0.7f, 0.2f)); // Sun
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset){
    auto* ctx =
        static_cast<SimulationContext*>(glfwGetWindowUserPointer(window));

    Engine *engine = ctx->engine;

    // Scroll up = zoom in
    engine->fov -= (float)yoffset;

    // Clamp zoom range
    if(engine->fov < 5.0f) engine->fov = 5.0f;
    if(engine->fov > 90.f) engine->fov = 90.f;
}



// Vertix drawing uses floats and not doubles
glm::vec3 spherical2carteesian(float radius, float theta, float phi, double x, double y, double z) {
    glm::vec3 vertex_coord;

    vertex_coord.x = (float)x + radius*sin(theta)*cos(phi);
    vertex_coord.y = (float)y + radius*sin(theta)*sin(phi);
    vertex_coord.z = (float)z + radius*cos(theta);

    return vertex_coord;
}


// Calculate y-height for grid with curvature math
float yCurvature(float x, float z, const std::vector<Celestial_Objects> &objects) {
    double totalY = 0.0;

    for(const auto &obj : objects) {
        double dx = x - obj.x;
        double dz = z - obj.z;

        double r = sqrt(dx*dx + dz*dz);

        r = std::max(r, obj.r_co);

        double rs = (2.0 * G_SIM * obj.mass) / (c*c);

        totalY +=  - 5e7 * sqrt(rs / r);
    }

    return (float) totalY;
}

// Draw gravity curvature grid
std::vector<Vertex> createFlatGrid(float size, int divisions)
{
    std::vector<Vertex> vertices;

    glm::vec3 color(1.0f);

    float step = size / divisions;

    for (int z = 0; z <= divisions; ++z)
    {
        for (int x = 0; x < divisions; ++x)
        {
            float x0 = -size / 2.0f + x * step;
            float x1 = x0 + step;

            float zPos = -size / 2.0f + z * step;

            vertices.push_back({glm::vec3(x0, 0.0f, zPos), color});
            vertices.push_back({glm::vec3(x1, 0.0f, zPos), color});
        }
    }

    for (int x = 0; x <= divisions; ++x)
    {
        for (int z = 0; z < divisions; ++z)
        {
            float z0 = -size / 2.0f + z * step;
            float z1 = z0 + step;

            float xPos = -size / 2.0f + x * step;

            vertices.push_back({glm::vec3(xPos, 0.0f, z0), color});
            vertices.push_back({glm::vec3(xPos, 0.0f, z1), color});
        }
    }

    return vertices;
}

void focusCameraOnObject(Engine &engine, Celestial_Objects &obj) {

    glm::vec3 target(obj.x, obj.y, obj.z);

    glm::vec3 offset(0.0f, 0.0f, obj.r_render*2.02f); // camera distance behind object

    engine.cameraPos = target + offset;

    engine.cameraFront = glm::normalize(target - engine.cameraPos);
}

void syncFreeCameraFromOrbit(Engine& engine)
{
    // Current forward vector already points correctly
    glm::vec3 front = glm::normalize(engine.cameraFront);

    // Convert direction vector -> yaw/pitch
    engine.yaw = glm::degrees(atan2(front.z, front.x));

    engine.pitch = glm::degrees(asin(front.y));

    // Prevent precision weirdness
    if (engine.pitch > 89.0f) engine.pitch = 89.0f;
    if (engine.pitch < -89.0f) engine.pitch = -89.0f;
}


// Resets the simulation when R is pressed
void simulationReset(std::vector<Celestial_Objects> &objects, std::vector<Celestial_Objects> &objectsCopy) {
    objects.clear();
    objects = objectsCopy;
}

// Planet constructor
void addPlanet(std::vector<Celestial_Objects>& objects, double distance, double mass, double planetRadius, double renderRadius, glm::vec3 color, double phase = 0.0)
{
    // Distance is away from sun
    double x = distance * cos(phase);
    double z = distance * sin(phase);

    // Places Sun at centre
    if(distance == 0) {
        objects.emplace_back(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), mass, planetRadius, renderRadius, color);
    }

    // Places planets
    else {
        double v = sqrt(G_SIM / distance);

        double vx = -v * sin(phase);
        double vz =  v * cos(phase);

        objects.emplace_back(vec3(x, 0.0, z), vec3(vx, 0.0, vz), mass, planetRadius, renderRadius, color);
    }

}

// Moon constructor
void addMoon(std::vector<Celestial_Objects>& objects, Celestial_Objects & planet, double distance, double mass, double planetRadius, double renderRadius, glm::vec3 color, double phase = 0.0)
{
    // Distance is away from specified planet
    double x = planet.x + distance * cos(phase);
    double z = planet.z +  distance * sin(phase);

    if(distance == 0) {
        objects.emplace_back(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), mass, planetRadius, renderRadius, color);
    }
    else {
        double v = sqrt(G_SIM * planet.mass / distance);

        double vx = planet.vx - v * sin(phase);
        double vz = planet.vz + v * cos(phase);

        objects.emplace_back(vec3(x, 0.0, z), vec3(vx, 0.0, vz), mass, planetRadius, renderRadius, color);
    }

}

// Generates the grid with GPU
void renderGrid(Shader &shader, std::vector<Celestial_Objects> &objects) {
for (int i = 0; i < objects.size(); i++)
{
    std::string posName =
        "objectPositions[" + std::to_string(i) + "]";

    glUniform3f(
        glGetUniformLocation(shader.ID, posName.c_str()),
        objects[i].x,
        objects[i].y,
        objects[i].z
    );

    std::string massName =
        "objectMasses[" + std::to_string(i) + "]";

    glUniform1f(
        glGetUniformLocation(shader.ID, massName.c_str()),
        objects[i].mass
    );

    std::string radiusName =
        "objectRadii[" + std::to_string(i) + "]";

    glUniform1f(
        glGetUniformLocation(shader.ID, radiusName.c_str()),
        objects[i].r_co
    );
}

glUniform1i(
    glGetUniformLocation(shader.ID, "objectCount"),
    objects.size()
);

}

std::vector<Vertex> createSphereMesh(int res)
{
    std::vector<Vertex> vertices;

    // Unit sphere radius = 1
    float radius = 1.0f;

    for(int i = 0; i < res; i++)
    {
        float theta1 = (i / (float)res) * M_PI;
        float theta2 = ((i + 1) / (float)res) * M_PI;

        for(int j = 0; j < res; j++)
        {
            float phi1 = (j / (float)res) * 2.0f * M_PI;
            float phi2 = ((j + 1) / (float)res) * 2.0f * M_PI;

            glm::vec3 v1(
                radius * sin(theta1) * cos(phi1),
                radius * cos(theta1),
                radius * sin(theta1) * sin(phi1)
            );

            glm::vec3 v2(
                radius * sin(theta1) * cos(phi2),
                radius * cos(theta1),
                radius * sin(theta1) * sin(phi2)
            );

            glm::vec3 v3(
                radius * sin(theta2) * cos(phi1),
                radius * cos(theta2),
                radius * sin(theta2) * sin(phi1)
            );

            glm::vec3 v4(
                radius * sin(theta2) * cos(phi2),
                radius * cos(theta2),
                radius * sin(theta2) * sin(phi2)
            );

            glm::vec3 color(1.0f);

            // Triangle 1
            vertices.push_back({v1, color});
            vertices.push_back({v2, color});
            vertices.push_back({v3, color});

            // Triangle 2
            vertices.push_back({v3, color});
            vertices.push_back({v2, color});
            vertices.push_back({v4, color});
        }
    }

    return vertices;
}

// Renders the spheres with GPU
void renderSpheres(Shader &shader, std::vector<Vertex> sphereVertices, std::vector<Celestial_Objects> &objects, GLuint & sphereVAO) {
for(auto &obj : objects) {
    glm::mat4 model = glm::mat4(1.0f);

    // Move sphere into world
    model = glm::translate(
        model,
        glm::vec3(obj.x, obj.y, obj.z)
    );

    // Scale sphere radius
    model = glm::scale(
        model,
        glm::vec3(obj.r_render)
    );

    shader.setMat4("model", model);

    shader.setVec3("objectColor", obj.color);

    glBindVertexArray(sphereVAO);

    glDrawArrays(GL_TRIANGLES, 0, sphereVertices.size());
}
}



int main() {
    // Define Engine here to work with screen height/width during collisions
    Engine engine;

    // Define the shaders
    Shader sphereShader(sphereShaderSource, fragmentShaderSource, true);
    Shader gridShader(gridShaderSource, fragmentShaderSource, true);

    // Define the celestial objects and a copy
    std::vector<Celestial_Objects> objects;
    std::vector<Celestial_Objects> objectsCopy;


    // Add the objects
    addPlanet(objects, 0.0, 1.0, 0.005, 0.05, vec3(1.0f, 0.7f, 0.2f)); // Sun
    addPlanet(objects, 0.387, 1.66e-7, 1.63e-5, 0.0001, vec3(0.65f, 0.62f, 0.58f)); // Mercury
    addPlanet(objects, 0.723, 3.003e-6, 4.26352e-5, 0.0008, vec3(0.9f, 0.8f, 0.4f)); // Venus
    addPlanet(objects, 1.000, 3.003e-6, 4.26352e-5, 0.001, vec3(0.0f, 0.7f, 1.0f)); // Earth
    addPlanet(objects, 1.524, 3.2e-7, 3.2e-5, 0.0008, vec3(1.0f, 0.3f, 0.2f));  // Mars
    addPlanet(objects, 5.203, 9.54e-4, 4.67e-4, 0.002, vec3(0.9f, 0.8f, 0.6f)); // Jupiter
    addPlanet(objects, 9.537, 2.86e-4, 3.89e-4, 0.008, vec3(0.9f, 0.8f, 0.7f)); //Saturn
    addPlanet(objects, 19.2, 4.37e-5, 1.7e-4, 0.006, vec3(0.5f, 0.8f, 1.0f)); //Uranus
    addPlanet(objects, 30.05, 5.15e-5, 1.65e-4, 0.006, vec3(0.2f, 0.4f, 1.0f)); //Neptune
    

    addMoon(objects, objects[3], 0.00257, 3.7e-8, 1.16e-5, 0.00005, vec3(0.9f, 0.9f, 0.9f)); // Earth Moon
    addMoon(objects, objects[5], 0.00282, 4.7e-8, 1e-5, 0.0002, vec3(1.0f, 0.9f, 0.5f));
    addMoon(objects, objects[5], 0.00449, 2.5e-8, 1e-5, 0.00018, vec3(0.8f, 0.8f, 1.0f));

    objectsCopy = objects;

    // Storage vector for all celestial positions
    std::vector<Vertex> sphereVertices = createSphereMesh(20);
    std::vector<Vertex> gridVertices = createFlatGrid(80.0f, 800);

    // Create VAO/VBO for spheres and grids
    GLuint sphereVAO, sphereVBO;
    GLuint gridVAO, gridVBO;
    
    glGenVertexArrays(1, &sphereVAO);
    glGenBuffers(1, &sphereVBO);

    glGenVertexArrays(1, &gridVAO);
    glGenBuffers(1, &gridVBO);

    glLineWidth(2.0f);

    // For spawning planets
    auto* ctx = new SimulationContext();
    ctx->engine = &engine;
    ctx->objects = &objects;

    glfwSetWindowUserPointer(engine.window, ctx);

    glfwSetMouseButtonCallback(engine.window, mouse_button_callback);

    // Send grid data to GPU
    runVAOVBO(gridVertices, gridVAO, gridVBO);
    runVAOVBO(sphereVertices, sphereVAO, sphereVBO);

    // Time step (day)
    float dt =  1.0f/365.0f;       
    float dlambda = dt * 0.02;
    
    while(!glfwWindowShouldClose(engine.window)) {

        glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        gravityStep(objects, dlambda);

        // render spheres
        sphereShader.use();
        engine.run(sphereShader, objects, objectsCopy);
        renderSpheres(sphereShader, sphereVertices, objects, sphereVAO);

        // render grid
        gridShader.use();
        renderGrid(gridShader, objects);
        engine.run(gridShader, objects, objectsCopy);
        glBindVertexArray(gridVAO);
        glDrawArrays(GL_LINES, 0, gridVertices.size());

        glfwSwapBuffers(engine.window);
    }

    return 0;
}

