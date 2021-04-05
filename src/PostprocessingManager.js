import * as THREE from "three";
//Shaders
import { Shaders } from "./Shader.js";

export class PostprocessingManager {
  constructor(scene, camera, renderer) {
    this.renderTarget = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight
    );
    this.renderTarget.depthBuffer = true;
    this.renderTarget.depthTexture = new THREE.DepthTexture();
    this.glowTarget = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight
    );

    this.pass1 = new THREE.WebGLRenderTarget(
      window.innerWidth / 2,
      window.innerHeight / 2
    );
    this.pass2 = new THREE.WebGLRenderTarget(
      window.innerWidth / 2,
      window.innerHeight / 2
    );
    this.pass3 = new THREE.WebGLRenderTarget(
      window.innerWidth / 2,
      window.innerHeight / 2
    );
    this.pass4 = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight
    );
    this.pass5 = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight
    );
    this.pass6 = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight
    );

    this.posttarget = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight
    );

    this.bloomTarget = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight
    );
    this.dofTarget = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight
    );

    var pars = {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat
    };
    this.renderTargetsHorizontal = [];
    this.renderTargetsVertical = [];
    this.nMips = 5;
    this.resolution = new THREE.Vector2(window.innerWidth, window.innerHeight);
    var resx = Math.round(this.resolution.x / 2);
    var resy = Math.round(this.resolution.y / 2);

    for (var i = 0; i < this.nMips; i++) {
      var renderTargetHorizonal = new THREE.WebGLRenderTarget(resx, resy, pars);

      renderTargetHorizonal.texture.name = "UnrealBloomPass.h" + i;
      renderTargetHorizonal.texture.generateMipmaps = false;

      this.renderTargetsHorizontal.push(renderTargetHorizonal);

      var renderTargetVertical = new THREE.WebGLRenderTarget(resx, resy, pars);

      renderTargetVertical.texture.name = "UnrealBloomPass.v" + i;
      renderTargetVertical.texture.generateMipmaps = false;

      this.renderTargetsVertical.push(renderTargetVertical);

      resx = Math.round(resx / 2);

      resy = Math.round(resy / 2);
    }

    this.separableBlurMaterials = [];
    var kernelSizeArray = [3, 5, 7, 9, 11];
    var resx = Math.round(this.resolution.x / 2);
    var resy = Math.round(this.resolution.y / 2);

    for (var i = 0; i < this.nMips; i++) {
      this.separableBlurMaterials.push(
        this.getSeperableBlurMaterial(kernelSizeArray[i])
      );

      this.separableBlurMaterials[i].uniforms[
        "texSize"
      ].value = new THREE.Vector2(resx, resy);

      resx = Math.round(resx / 2);

      resy = Math.round(resy / 2);
    }
    this.compositeMaterial = this.getCompositeMaterial(this.nMips);
    this.compositeMaterial.uniforms[
      "blurTexture1"
    ].value = this.renderTargetsVertical[0].texture;
    this.compositeMaterial.uniforms[
      "blurTexture2"
    ].value = this.renderTargetsVertical[1].texture;
    this.compositeMaterial.uniforms[
      "blurTexture3"
    ].value = this.renderTargetsVertical[2].texture;
    this.compositeMaterial.uniforms[
      "blurTexture4"
    ].value = this.renderTargetsVertical[3].texture;
    this.compositeMaterial.uniforms[
      "blurTexture5"
    ].value = this.renderTargetsVertical[4].texture;
    this.compositeMaterial.uniforms["bloomStrength"].value = 1.5;
    this.compositeMaterial.uniforms["bloomRadius"].value = 1.0;
    this.compositeMaterial.needsUpdate = true;

    var bloomFactors = [1.0, 0.8, 0.6, 0.4, 0.2];
    this.compositeMaterial.uniforms["bloomFactors"].value = bloomFactors;
    this.bloomTintColors = [
      new THREE.Vector3(1, 1, 1),
      new THREE.Vector3(1, 1, 1),
      new THREE.Vector3(1, 1, 1),
      new THREE.Vector3(1, 1, 1),
      new THREE.Vector3(1, 1, 1)
    ];
    this.compositeMaterial.uniforms[
      "bloomTintColors"
    ].value = this.bloomTintColors;

    // this.renderTarget = new THREE.WebGLRenderTarget(window.innerWidth, window.innerHeight);
    // this.renderTarget.samples = 2;

    console.log(renderer.capabilities.isWebGL2);
    //*************************************** */

    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;

    let shaders = new Shaders();

    let FXAAuniforms = {
      tDiffuse: {
        value: this.bloomTarget.texture
      },
      resolution: {
        value: new THREE.Vector2()
      }
    };

    let uniforms = {
      tDiffuse: {
        value: this.bloomTarget.texture
      },
      em: {
        value: this.pass4.texture
      }
    };

    let bloomuniforms = {
      tDiffuse: {
        value: this.posttarget.texture
      },
      em: {
        value: this.pass6.texture
      }
    };
    let dofuniforms = {
      original: {
        value: this.posttarget.texture
      },
      blurred: {
        value: this.pass4.texture
      },
      depth: {
        value: this.renderTarget.texture
      },
      dis: {
        value: 10
      }
    };

    let uniformsH = {
      tex: {
        value: this.glowTarget.texture
      },
      width: { value: window.innerWidth },
      height: { value: window.innerHeight }
    };

    let uniformsV = {
      tex: {
        value: this.glowTarget.texture
      },
      width: { value: window.innerWidth },
      height: { value: window.innerHeight }
    };

   

    let lightUniforms={
color:{
  value:[]
},pos:{
value:[]
}
    }
    this.scene.traverse((node) => {
      if (node instanceof THREE.Mesh) {
        
        if (node.material.name.charAt(0) == "E") {
         let n = new THREE.Vector3();
         node.getWorldPosition(n);
         n.y = -1.5
         lightUniforms.pos.value.push(n);
         let c = node.material.emissive
        lightUniforms.color.value.push(new THREE.Vector3(c.r,c.g,c.b))
        }
      }
    });
    lightUniforms.pos.value[0].x = 1.7;
   // lightUniforms.pos.value[0].y = 0.;
    lightUniforms.pos.value[0].z = -1.0;
    lightUniforms.pos.value[1].z = -0.5;
    lightUniforms.pos.value[1].x = -1.5;
    lightUniforms.pos.value[2].z = 0.5;
    

    this.lightShader = new THREE.ShaderMaterial({
     uniforms:lightUniforms,
      fragmentShader: shaders.lightShader(),
      vertexShader: shaders.vertexShader()
    });
    this.FXAA = new THREE.ShaderMaterial({
      uniforms: FXAAuniforms,
      fragmentShader: shaders.FXAA(),
      vertexShader: shaders.vertexShader()
    });
    this.VerticalBlur = new THREE.ShaderMaterial({
      uniforms: uniformsV,
      fragmentShader: shaders.Blurfs(),
      vertexShader: shaders.BlurVvs()
    });
    this.HorizontalBlur = new THREE.ShaderMaterial({
      uniforms: uniformsH,
      fragmentShader: shaders.Blurfs(),
      vertexShader: shaders.BlurHvs()
    });

    this.bloomMat = new THREE.ShaderMaterial({
      uniforms: bloomuniforms,
      fragmentShader: shaders.bloomMix(),
      vertexShader: shaders.vertexShader()
    });
    this.finalMat = new THREE.ShaderMaterial({
      uniforms: uniforms,
      fragmentShader: shaders.fragmentShader(),
      vertexShader: shaders.vertexShader()
    });

    this.dofMat = new THREE.ShaderMaterial({
      uniforms: dofuniforms,
      fragmentShader: shaders.dof(),
      vertexShader: shaders.vertexShader()
    });

    this.matdictionary = {};
    this.postScene = new THREE.Scene();
    this.postCamera = new THREE.OrthographicCamera(
      -1, // left
      1, // right
      1, // top
      -1, // bottom
      -1, // near,
      1 // far
    );

    const plane = new THREE.PlaneBufferGeometry(2, 2);
    this.quad = new THREE.Mesh(plane, this.finalMat);
    this.quad.position.z = -1;
    this.postScene.add(this.quad);
    this.timer = 0;
    this.sceneColor = this.scene.background
  }

  getSeperableBlurMaterial(kernelRadius) {
    return new THREE.ShaderMaterial({
      defines: {
        KERNEL_RADIUS: kernelRadius,
        SIGMA: kernelRadius
      },

      uniforms: {
        colorTexture: { value: null },
        texSize: { value: new THREE.Vector2(0.5, 0.5) },
        direction: { value: new THREE.Vector2(0.5, 0.5) }
      },

      vertexShader:
        "varying vec2 vUv;\n\
        void main() {\n\
          vUv = uv;\n\
          gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\
        }",

      fragmentShader:
        "#include <common>\
        varying vec2 vUv;\n\
        uniform sampler2D colorTexture;\n\
        uniform vec2 texSize;\
        uniform vec2 direction;\
        \
        float gaussianPdf(in float x, in float sigma) {\
          return 0.39894 * exp( -0.5 * x * x/( sigma * sigma))/sigma;\
        }\
        void main() {\n\
          vec2 invSize = 1.0 / texSize;\
          float fSigma = float(SIGMA);\
          float weightSum = gaussianPdf(0.0, fSigma);\
          vec3 diffuseSum = texture2D( colorTexture, vUv).rgb * weightSum;\
          for( int i = 1; i < KERNEL_RADIUS; i ++ ) {\
            float x = float(i);\
            float w = gaussianPdf(x, fSigma);\
            vec2 uvOffset = direction * invSize * x;\
            vec3 sample1 = texture2D( colorTexture, vUv + uvOffset).rgb;\
            vec3 sample2 = texture2D( colorTexture, vUv - uvOffset).rgb;\
            diffuseSum += (sample1 + sample2) * w;\
            weightSum += 2.0 * w;\
          }\
          gl_FragColor = vec4(diffuseSum/weightSum, 1.0);\n\
        }"
    });
  }

  resize() {
    this.renderTarget.setSize(window.innerWidth, window.innerHeight);
    this.glowTarget.setSize(window.innerWidth, window.innerHeight);
    this.bloomTarget.setSize(window.innerWidth, window.innerHeight);
    this.pass1.setSize(window.innerWidth, window.innerHeight);
    this.pass2.setSize(window.innerWidth, window.innerHeight);
    this.pass3.setSize(window.innerWidth, window.innerHeight);

    var resx = Math.round(window.innerWidth / 2);
    var resy = Math.round(window.innerWidth / 2);

    for (var i = 0; i < this.nMips; i++) {
      this.renderTargetsHorizontal[i].setSize(resx, resy);
      this.renderTargetsVertical[i].setSize(resx, resy);

      this.separableBlurMaterials[i].uniforms[
        "texSize"
      ].value = new THREE.Vector2(resx, resy);

      resx = Math.round(resx / 2);
      resy = Math.round(resy / 2);
    }
  }

  update(delta) {
    //Default scene render
    this.scene.background = this.sceneColor;
    this.renderer.setRenderTarget(this.renderTarget);
    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(null);
  
    this.scene.traverse((node) => {
      if (node instanceof THREE.Mesh) {
        if (node.material.name.charAt(0) == "E") {
         // console.log(node.material);

       //  this.lightShader.uniforms.pos = 
        }
      }
    });


    //emissive scene render
    this.scene.background = new THREE.Color("black");
    this.renderer.setRenderTarget(this.glowTarget);
    this.matdictionary = {};
    this.scene.traverse((node) => {
      if (node instanceof THREE.Mesh) {
        if (node.material.name.charAt(0) != "E") {
          // let co = new THREE.Color();
          // node.material.emissive.copy
          // if (cc.r>2.) {
          //console.log(node.material);
          let n = node.material.name;
          this.matdictionary[n] = node.material;
          //node.material = new THREE.MeshBasicMaterial({ color: 0x000000 });
          node.material = this.lightShader.clone();
          node.material.name = n;
          //this.scene.remove(node);
        }
      }
    });


    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(null);
    this.scene.traverse((node) => {
      if (node instanceof THREE.Mesh) {
        if (node.material.name.charAt(0) != "E") {
          //console.log(node.material);
          //let n = node.material.name;
          //this.matdictionary[n] = node.material;
          node.material = this.matdictionary[node.material.name];
          //node.material.name = n;
          //this.scene.remove(node);
        }
      }
    });

    var inputRenderTarget = this.glowTarget;

    for (var i = 0; i < this.nMips; i++) {
      this.quad.material = this.separableBlurMaterials[i];

      this.separableBlurMaterials[i].uniforms["colorTexture"].value =
        inputRenderTarget.texture;
      this.separableBlurMaterials[i].uniforms[
        "direction"
      ].value = new THREE.Vector2(1.0, 0.0);
      this.renderer.setRenderTarget(this.renderTargetsHorizontal[i]);
      this.renderer.clear();
      this.renderer.render(this.postScene, this.postCamera);
      this.renderer.setRenderTarget(null);

      this.separableBlurMaterials[i].uniforms[
        "colorTexture"
      ].value = this.renderTargetsHorizontal[i].texture;
      this.separableBlurMaterials[i].uniforms[
        "direction"
      ].value = new THREE.Vector2(0.0, 1.0);
      this.renderer.setRenderTarget(this.renderTargetsVertical[i]);
      this.renderer.clear();
      this.renderer.render(this.postScene, this.postCamera);
      this.renderer.setRenderTarget(null);

      inputRenderTarget = this.renderTargetsVertical[i];
    }

    this.quad.material = this.compositeMaterial;
    this.compositeMaterial.uniforms["bloomStrength"].value = 1.5;
    this.compositeMaterial.uniforms["bloomRadius"].value = 1.0;
    this.compositeMaterial.uniforms[
      "bloomTintColors"
    ].value = this.bloomTintColors;

    this.renderer.setRenderTarget(this.renderTargetsHorizontal[0]);
    this.renderer.clear();
    this.renderer.render(this.postScene, this.postCamera);
    this.renderer.setRenderTarget(null);

    let t = 8;

    //pass 2
    this.renderer.setRenderTarget(this.pass1);
    this.quad.material = this.VerticalBlur;
    this.quad.material.uniforms.height.value = window.innerHeight / t;
    this.quad.material.uniforms.tex.value = this.renderTarget.texture;
    this.renderer.render(this.postScene, this.postCamera);
    this.renderer.setRenderTarget(null);
    t /= 2.0;
    this.renderer.setRenderTarget(this.pass2);
    this.quad.material = this.HorizontalBlur;
    this.quad.material.uniforms.width.value = window.innerWidth / t;
    this.quad.material.uniforms.tex.value = this.pass1.texture;
    this.renderer.render(this.postScene, this.postCamera);
    this.renderer.setRenderTarget(null);
    t /= 2.0;
    this.renderer.setRenderTarget(this.pass3);
    this.quad.material = this.VerticalBlur;
    this.quad.material.uniforms.height.value = window.innerHeight / t;
    this.quad.material.uniforms.tex.value = this.pass2.texture;
    this.renderer.render(this.postScene, this.postCamera);
    this.renderer.setRenderTarget(null);
    t /= 2.0;
    this.renderer.setRenderTarget(this.pass4);
    this.quad.material = this.HorizontalBlur;
    this.quad.material.uniforms.width.value = window.innerWidth / t;
    this.quad.material.uniforms.tex.value = this.pass3.texture;
    this.renderer.render(this.postScene, this.postCamera);
    this.renderer.setRenderTarget(null);

    this.renderer.setRenderTarget(this.dofTarget);
    this.quad.material = this.bloomMat;
    this.quad.material = this.dofMat;
    this.quad.material.uniforms.original.value = this.renderTarget.texture;
    this.quad.material.uniforms.blurred.value = this.pass4.texture;
    this.quad.material.uniforms.depth.value = this.renderTarget.depthTexture;
    let pos = new THREE.Vector3(0, -2.0, 0);
    let k = this.camera.position.distanceTo(pos);

    this.quad.material.uniforms.dis.value = k;
    this.renderer.render(this.postScene, this.postCamera);
    this.renderer.setRenderTarget(null);

    this.renderer.setRenderTarget(this.bloomTarget);
    this.quad.material = this.bloomMat;
    this.quad.material.uniforms.tDiffuse.value = this.dofTarget.texture;
    this.quad.material.uniforms.em.value = this.renderTargetsHorizontal[0].texture;
    this.renderer.render(this.postScene, this.postCamera);
    this.renderer.setRenderTarget(null);

    this.renderer.setRenderTarget(this.posttarget);
    this.quad.material = this.finalMat;
    this.quad.material.uniforms.tDiffuse.value = this.bloomTarget.texture;
    this.renderer.render(this.postScene, this.postCamera);
    this.renderer.setRenderTarget(null);
    var pixelRatio = this.renderer.getPixelRatio();
    this.quad.material = this.FXAA;
    this.quad.material.uniforms.tDiffuse.value = this.posttarget.texture;
    this.quad.material.uniforms.resolution.value.x =
      1 / (window.innerWidth * pixelRatio);
    this.quad.material.uniforms.resolution.value.y =
      1 / (window.innerHeight * pixelRatio);
    this.renderer.render(this.postScene, this.postCamera);
  }

  getSeperableBlurMaterial(kernelRadius) {
    return new THREE.ShaderMaterial({
      defines: {
        KERNEL_RADIUS: kernelRadius,
        SIGMA: kernelRadius
      },

      uniforms: {
        colorTexture: { value: null },
        texSize: { value: new THREE.Vector2(0.5, 0.5) },
        direction: { value: new THREE.Vector2(0.5, 0.5) }
      },

      vertexShader:
        "varying vec2 vUv;\n\
        void main() {\n\
          vUv = uv;\n\
          gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\
        }",

      fragmentShader:
        "#include <common>\
        varying vec2 vUv;\n\
        uniform sampler2D colorTexture;\n\
        uniform vec2 texSize;\
        uniform vec2 direction;\
        \
        float gaussianPdf(in float x, in float sigma) {\
          return 0.39894 * exp( -0.5 * x * x/( sigma * sigma))/sigma;\
        }\
        void main() {\n\
          vec2 invSize = 1.0 / texSize;\
          float fSigma = float(SIGMA);\
          float weightSum = gaussianPdf(0.0, fSigma);\
          vec3 diffuseSum = texture2D( colorTexture, vUv).rgb * weightSum;\
          for( int i = 1; i < KERNEL_RADIUS; i ++ ) {\
            float x = float(i);\
            float w = gaussianPdf(x, fSigma);\
            vec2 uvOffset = direction * invSize * x;\
            vec3 sample1 = texture2D( colorTexture, vUv + uvOffset).rgb;\
            vec3 sample2 = texture2D( colorTexture, vUv - uvOffset).rgb;\
            diffuseSum += (sample1 + sample2) * w;\
            weightSum += 2.0 * w;\
          }\
          gl_FragColor = vec4(diffuseSum/weightSum, 1.0);\n\
        }"
    });
  }

  getCompositeMaterial(nMips) {
    return new THREE.ShaderMaterial({
      defines: {
        NUM_MIPS: nMips
      },

      uniforms: {
        blurTexture1: { value: null },
        blurTexture2: { value: null },
        blurTexture3: { value: null },
        blurTexture4: { value: null },
        blurTexture5: { value: null },
        dirtTexture: { value: null },
        bloomStrength: { value: 1.0 },
        bloomFactors: { value: null },
        bloomTintColors: { value: null },
        bloomRadius: { value: 1.0 }
      },

      vertexShader:
        "varying vec2 vUv;\n\
          void main() {\n\
            vUv = uv;\n\
            gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\
          }",

      fragmentShader:
        "varying vec2 vUv;\
          uniform sampler2D blurTexture1;\
          uniform sampler2D blurTexture2;\
          uniform sampler2D blurTexture3;\
          uniform sampler2D blurTexture4;\
          uniform sampler2D blurTexture5;\
          uniform sampler2D dirtTexture;\
          uniform float bloomStrength;\
          uniform float bloomRadius;\
          uniform float bloomFactors[NUM_MIPS];\
          uniform vec3 bloomTintColors[NUM_MIPS];\
          \
          float lerpBloomFactor(const in float factor) { \
            float mirrorFactor = 1.2 - factor;\
            return mix(factor, mirrorFactor, bloomRadius);\
          }\
          \
          void main() {\
            gl_FragColor = bloomStrength * ( lerpBloomFactor(bloomFactors[0]) * vec4(bloomTintColors[0], 1.0) * texture2D(blurTexture1, vUv) + \
                             lerpBloomFactor(bloomFactors[1]) * vec4(bloomTintColors[1], 1.0) * texture2D(blurTexture2, vUv) + \
                             lerpBloomFactor(bloomFactors[2]) * vec4(bloomTintColors[2], 1.0) * texture2D(blurTexture3, vUv) + \
                             lerpBloomFactor(bloomFactors[3]) * vec4(bloomTintColors[3], 1.0) * texture2D(blurTexture4, vUv) + \
                             lerpBloomFactor(bloomFactors[4]) * vec4(bloomTintColors[4], 1.0) * texture2D(blurTexture5, vUv) );\
                            }"
    });
  }
}
