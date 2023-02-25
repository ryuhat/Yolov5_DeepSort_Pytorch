// Create a scene and a camera
var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera( 75, window.innerWidth/window.innerHeight, 0.1, 1000 );

// Create a renderer
var renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );

// Load the data from the CSV file
var loader = new THREE.FileLoader();
loader.load( '733.txt', function ( data ) {

  // Parse the CSV data
  var lines = data.split( '\n' );
  var trajectories = {};
  for ( var i = 0; i < lines.length; i ++ ) {
    var line = lines[ i ].split( ' ' );
    var id = line[ 0 ];
    if ( ! trajectories[ id ] ) {
      trajectories[ id ] = {
        x: [],
        y: [],
        z: []
      };
    }
    trajectories[ id ].x.push( parseFloat( line[ 2 ] ) );
    trajectories[ id ].y.push( parseFloat( line[ 3 ] ) );
    trajectories[ id ].z.push( parseFloat( line[ 4 ] ) );
  }

  // Create a line for each trajectory
  for ( var id in trajectories ) {
    var geometry = new THREE.BufferGeometry().setFromPoints( 
      trajectories[ id ].x.map( ( x, i ) => new THREE.Vector3( x, trajectories[ id ].y[ i ], trajectories[ id ].z[ i ] ) ) 
    );
    var material = new THREE.LineBasicMaterial( { color: Math.random() * 0xffffff } );
    var line = new THREE.Line( geometry, material );
    scene.add( line );
  }

  // Set the camera position
  camera.position.z = 10;

  // Render the scene
  function animate() {
    requestAnimationFrame( animate );
    renderer.render( scene, camera );
  }
  animate();

} );
