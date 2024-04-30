import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TexturedPlane, setup_renderer, setup_lighting } from './helper.js';
import { setup_sky } from './sky.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';

let sun = new THREE.Vector3();

const renderer = setup_renderer();
const sky = setup_sky(sun);
const [directional_light, ambiant_light] = setup_lighting(sun);

// Camera
const orbit_camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 2000);
orbit_camera.up.set(0, 0, 1);
const controls = new OrbitControls(orbit_camera, renderer.domElement);
orbit_camera.position.set(-200, -200, 200);
controls.update();

let camera = orbit_camera;
let possessed_camera = false;

// Scene
const scene = new THREE.Scene();

const table = new TexturedPlane("data/table.png", 0, 0, 0, 300, 200);
table.setReceiveShadow(true);

// OAK camera (4056x3040)
// const oak_camera = new THREE.PerspectiveCamera(74, 4056 / 3040, 1, 1000);    // OAK-1 W
const oak_camera = new THREE.PerspectiveCamera(55, 4056 / 3040, 1, 1000);   // OAK-D
oak_camera.position.set(0, -100, 160);
oak_camera.rotateX(30 * Math.PI / 180);
const oak_helper = new THREE.CameraHelper(oak_camera);
oak_camera.updateWorldMatrix();
// const directional_light_helper = new THREE.DirectionalLightHelper(directional_light, 5);
const axes_helper = new THREE.AxesHelper(100);

// Lighting
scene.add(directional_light);
scene.add(ambiant_light);
scene.add(sky);
// Helpers
// scene.add(directional_light_helper);
scene.add(oak_helper);
scene.add(axes_helper);
// Objects
scene.add(table.mesh);

// Controls
const gui = new GUI();

const folderCamera = gui.addFolder('Camera');
const propsLocal = {

    get 'Possess'() {

        return possessed_camera;

    },
    set 'Possess'(v) {

        possessed_camera = v;
        camera = v ? oak_camera : orbit_camera;
        oak_helper.visible = !v;
        controls.enabled = !v;

    },
};

folderCamera.add(propsLocal, 'Possess');

const rad2deg = (x) => {
    return 180*x/Math.PI;
}

const deg2rad = (x) => {
    return Math.PI*x/180;
}

const update_camera_position = () => {
    fetch("http://localhost:8765/get_position").then((response) => {
        return JSON.parse(response.statusText);
    }).then((data) => {
        const x = data[0] * 100
        const y = data[1] * 100
        const z = data[2] * 100
        // console.log("Set position to", x, y, z);
        oak_camera.position.set(x, y, z);
        oak_camera.updateWorldMatrix();
    }).catch((error) => {
        console.log("no answer");
    })
    fetch("http://localhost:8765/get_rotation").then((response) => {
        return JSON.parse(response.statusText);
    }).then((data) => {
        const x = data[0]
        const y = data[1]
        const z = data[2]
        oak_camera.setRotationFromEuler(new THREE.Euler(x, -y, -z));
        // oak_camera.rotateX(x);
        // oak_camera.rotateY(y);
        // oak_camera.rotateZ(z);
        oak_camera.updateWorldMatrix();
        console.log("rotation", x, y, z);
    }).catch((error) => {
        console.log("no answer for rotation");
    })
}

setInterval(() => {
    update_camera_position();
}, 500);

function update(t) {
    const show_texture = Math.cos(Math.PI * t) > 0;
    // table.plane_material.map = show_texture ? table.texture : null;
    table.plane_material.map = table.texture;
}

function animate(t) {
    requestAnimationFrame(animate);

    update(t / 1000);

    controls.update();
    renderer.render(scene, camera);
}

animate();