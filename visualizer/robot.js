import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

class BasicObject {
    constructor() {
        this.mesh = undefined;
    }

    setup_shadows() {
        this.mesh.castShadow = true;
        this.mesh.receiveShadow = true;
    }

    set_position(x, y) {
        this.mesh.position.x = x;
        this.mesh.position.z = y;
    }

    set_rotation(alpha) {
        this.mesh.rotation.y = alpha;
    }

    register(scene) {
        scene.add(this.mesh);
    }
}

class Robot extends BasicObject {
    constructor(id) {
        super();
        this.id = id;

        const loader = new GLTFLoader();
        this.ready = false;

        let self = this;
        this.mesh = undefined;
        this.ready_callback = undefined;
        loader.load('public/R0.glb', function (gltf) {
            console.log(gltf.scene.children[0]);
            const mesh = gltf.scene.children[0];
            self.mesh = mesh;
            self.ready = true;
            self.setup_shadows();
            if (self.ready_callback) {
                self.ready_callback();
            }
        }, undefined, function (error) {
            console.error(error);
        });
    }

    setup_shadows() {
        for (const index in this.mesh.children) {
            if (Object.hasOwnProperty.call(this.mesh.children, index)) {
                const child = this.mesh.children[index];
                child.castShadow = true;
                // child.receiveShadow = true;        
            }
        }
    }

    set_position(x, y) {
        if (this.ready) {
            super.set_position(x, y);
        }
    }

    set_rotation(alpha) {
        if (this.ready) {
            super.set_rotation(alpha);
        }
    }

    register(scene) {
        if (this.ready) {
            console.log("Registered robot", this.id);
            scene.add(this.mesh);
        } else {
            this.ready_callback = () => { this.register(scene) };
        }
    }
}

class Marker extends BasicObject {
    constructor(id) {
        super();
        this.texture = new THREE.TextureLoader().load("public/marker_"+id+".png");
        this.marker_material = new THREE.MeshStandardMaterial({ map: this.texture });
        this.pole_material = new THREE.MeshStandardMaterial({ color: 0x333333 });
        this.marker_material.side = THREE.DoubleSide;
        const marker_size = 20; // centimeters
        const marker_height = 15; // centimeters
        this.marker_geometry = new THREE.PlaneGeometry(marker_size, marker_size, 2, 2).translate(0, marker_height, 0);
        this.pole_geometry = new THREE.CylinderGeometry(.6, .6, marker_height - marker_size / 2, 8, 2, false).translate(0, (marker_height - marker_size / 2) / 2, 0);

        this.marker_mesh = new THREE.Mesh(this.marker_geometry, this.marker_material);
        this.pole_mesh = new THREE.Mesh(this.pole_geometry, this.pole_material);

        this.mesh = new THREE.Group();
        this.mesh.add(this.marker_mesh);
        this.mesh.add(this.pole_mesh);

        this.setup_shadows();
    }

    setup_shadows() {
        this.marker_mesh.castShadow = true;
        this.marker_mesh.receiveShadow = true;
        this.pole_mesh.castShadow = true;
        this.pole_mesh.receiveShadow = true;
    }
}

export { Robot, Marker };