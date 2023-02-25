      // Set up the scene
      var scene = new THREE.Scene();
      var camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
      );
      var renderer = new THREE.WebGLRenderer();
      renderer.setSize(window.innerWidth, window.innerHeight);
      document.body.appendChild(renderer.domElement);

      // Set up the data
      var data = [
        { x: 0, y: 0, z: 0 },
        { x: 1, y: 1, z: 1 },
        { x: 2, y: 2, z: 2 },
        { x: 3, y: 3, z: 3 },
        { x: 4, y: 4, z: 4 },
        { x: 5, y: 5, z: 5 },
      ];

      // Set up the geometry and material
      var geometry = new THREE.BufferGeometry();
      var vertices = [];
      for (var i = 0; i < data.length; i++) {
        vertices.push(data[i].x, data[i].y, data[i].z);
      }
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(vertices, 3)
      );
      var material = new THREE.LineBasicMaterial({
        color: 0xffffff,
        linewidth: 2,
      });

      // Create the line
      var line = new THREE.Line(geometry, material);
      scene.add(line);

      // Set up the camera and controls
      camera.position.z = 10;
      var controls = new THREE.OrbitControls(camera, renderer.domElement);

      // Render the scene
      function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }
      animate();