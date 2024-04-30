import * as THREE from 'three';

class TexturedPlane {
    constructor(texture_path, x, y, z, width, height) {
        this.texture = new THREE.TextureLoader().load(texture_path);
        this.plane_material = new THREE.MeshStandardMaterial({ map: this.texture, side: THREE.DoubleSide });
        this.plane_geometry = new THREE.PlaneGeometry(width, height, 2, 2).rotateZ(Math.PI);
        this.plane_geometry.receiveShadows = true;
        this.mesh = new THREE.Mesh(this.plane_geometry, this.plane_material);
    }

    setReceiveShadow(receive) {
        this.mesh.receiveShadow = receive;
    }
}

function setup_renderer() {
    const renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap; // default THREE.PCFShadowMap
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.5;

    return renderer;
}

function setup_lighting(sun_position) {
    const light = new THREE.DirectionalLight( 0xffffff, 0.5 );
    const ambiant_light = new THREE.AmbientLight( 0x9fb6b9 ); // soft white light
    light.position.set(sun_position.x, sun_position.y, sun_position.z);
    const offset = new THREE.Vector3(1, 1, 1);
    light.position.multiplyScalar(300);
    // Shadows
    light.castShadow = true; // default false
    light.shadow.camera.far = 500;
    light.shadow.camera.bottom = -200;
    light.shadow.camera.top = 200;
    light.shadow.camera.left = -300;
    light.shadow.camera.right = 300;
    // light.shadow.mapSize.width = 1024; // default = 512
    // light.shadow.mapSize.height = 1024; // default = 512
    return [ light, ambiant_light ];
}

export { TexturedPlane, setup_renderer, setup_lighting }