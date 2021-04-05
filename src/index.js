import {
  Scene,
  Color,
  PerspectiveCamera,
  WebGLRenderer,
  DirectionalLight,
  HemisphereLight,
  Vector3,
  Clock,
  LinearFilter
} from "three";
import * as THREE from "three";
import OrbitControls from "three-orbitcontrols";
import GLTFLoader from "three-gltf-loader";
import { PostprocessingManager } from "./PostprocessingManager.js";

let container;
let camera;
let renderer;
let scene;
let controls;
let postprocessingManager;
const clock = new Clock();

function init() {
  container = document.querySelector("#scene-container");

  // Creating the scene
  scene = new Scene();
  scene.background = new Color(0x050515);
  
  const geometry = new THREE.PlaneGeometry(32, 32, 32);
  const material = new THREE.MeshPhongMaterial({
    color: scene.background,
    // color:0xffffff,
    transparent: true,side:THREE.DoubleSide,
  });
  material.specular = new THREE.Color("black");
  material.reflectivity = 0;
  const plane = new THREE.Mesh(geometry, material);
  plane.receiveShadow = true;
  material.name = "plane";

  material.onBeforeCompile = function (shader) {
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", "#include <common>")
      // .replace('gl_FragColor = vec4( outgoingLight, diffuseColor.a );', 'if(outgoingLight.g>outgoingLight.b&&outgoingLight.b>outgoingLight.r){ gl_FragColor= texture2D( map, vUv );}else{gl_FragColor = vec4( outgoingLight, diffuseColor.a );}')
      .replace(
        "gl_FragColor = vec4( outgoingLight, diffuseColor.a );",
        " gl_FragColor = vec4(vec3(outgoingLight.rgb),clamp(1.-((outgoingLight.r+outgoingLight.g+outgoingLight.b)/3.),0.,1.))/2.;"
      )
      .replace(
        "gl_FragColor.rgb *= gl_FragColor.a;",
        "gl_FragColor.rgb *=  gl_FragColor.a;"
      );
    //child.material.userData.shader = shader;
    // ref_object.material.userData.shader = shader;
  };

  //material.name = "plan////"
  plane.position.set(0, -2.0, 0.0);
  plane.setRotationFromEuler(new THREE.Euler(3.1415 / 2, 3.1415*1, 0));
 // scene.add(plane);
  createCamera();
  createLights();
  loadModels();
  createControls();
  createRenderer();
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  
}

function createCamera() {
  const fov = 35;
  const aspect = container.clientWidth / container.clientHeight;
  const near = 0.1;
  const far = 1000;
  camera = new PerspectiveCamera(fov, aspect, near, far);
  camera.position.set(-0, -0, 10);
}

function createLights() {
  const mainLight = new DirectionalLight(0xffffff, 4);
  mainLight.castShadow = true;
  mainLight.position.set(0, 10, 10);
  mainLight.shadow.mapSize.width = 212; // default
  mainLight.shadow.mapSize.height = 212; // default
  mainLight.shadow.camera.near = 0.5; // default
  mainLight.shadow.camera.far = 50; // default

  // const helper = new THREE.DirectionalLightHelper( mainLight, 5 );
  // scene.add( helper );

  const hemisphereLight = new HemisphereLight(0xddeeff, 0x202020, 0.9);
 // hemisphereLight.position.z = 2;
  //hemisphereLight.castShadow = true;
  scene.add(mainLight, hemisphereLight);
}

function loadModels() {
  const loader = new GLTFLoader();

  const onLoad = (result, position) => {
    let gltfMesh = result.scene;

    gltfMesh.traverse((node) => {
      if (node instanceof THREE.Mesh) {
      //  console.log(node.material);
        node.castShadow = true;
        node.receiveShadow = false;
        if (node.material.name != "None") {
          node.material.fog = false;
          node.material.roughness = 0;
          //node.material.emissiveIntensity = 1.0;
          // node.material.lights = false;
          //console.log(node.material);
          //node.material = new THREE.MeshBasicMaterial({color:0x000000});
          //gltfMesh.remove(node);
        }
      }
    });
    const model = result.scene.children[0];
    model.position.copy(position);

    scene.add(model);
    postprocessingManager = new PostprocessingManager(scene, camera, renderer);
  renderer.setAnimationLoop(() => {
    update();
    render();
  });
  };

  const onProgress = (progress) => {};

  const storkPositio = new Vector3(0, -2.0, 0);
  loader.load(
    "/src/models/street.glb",
    (gltf) => onLoad(gltf, storkPositio),
    onProgress
  );
}

function createRenderer() {
  renderer = new WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.gammaFactor = 1.2;
  renderer.gammaOutput = true;
  renderer.physicallyCorrectLights = true;

  container.appendChild(renderer.domElement);
}

function createControls() {
  controls = new OrbitControls(camera, container);
}

function update() {}

function render() {
  const delta = clock.getDelta();
  postprocessingManager.update(delta);
}

init();

function onWindowResize() {
  camera.aspect = container.clientWidth / container.clientHeight;

  // Update camera frustum
  camera.updateProjectionMatrix();

  renderer.setSize(container.clientWidth, container.clientHeight);
  postprocessingManager.resize();
}
window.addEventListener("resize", onWindowResize, false);
