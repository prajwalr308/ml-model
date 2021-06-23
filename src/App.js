import logo from './logo.svg';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import { useEffect, useRef } from 'react';

import * as blazeface from '@tensorflow-models/blazeface';


function App() {
  const videoRef = useRef(null);
  var model, mask_model, ctx, videoWidth, videoHeight, canvas,result;
	const video = document.getElementById('video');
	const state = {
	  backend: 'webgl'
	};
  useEffect(() => {
    async function setupCamera() {
      const stream = await navigator.mediaDevices.getUserMedia({
          'audio': false,
          'video': { facingMode: 'user' },
      });
      videoRef.current.srcObject = stream;
        return new Promise((resolve) => {
          videoRef.current.onloadedmetadata = () => {
            resolve(video);
          };
      });
    }
  
    const renderPrediction = async () => {
      tf.engine().startScope()
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      //estimatefaces model takes in 4 parameter (1) video, returnTensors, flipHorizontal, and annotateBoxes
      const predictions = await model.estimateFaces(video, true,false,false);
      const offset = tf.scalar(127.5);
      //check if prediction length is more than 0
      if (predictions.length > 0) {
        //clear context
          
          for (let i = 0; i < predictions.length; i++) {
            var text=""
            var start = predictions[i].topLeft.arraySync();
            var end = predictions[i].bottomRight.arraySync();
            var size = [end[0] - start[0], end[1] - start[1]];
            if(videoWidth<end[0] && videoHeight<end[0]){
              console.log("image out of frame")
              continue
            }
            var inputImage = tf.browser.fromPixels(video).toFloat()
            inputImage = inputImage.sub(offset).div(offset);
            inputImage=inputImage.slice([parseInt(start[1]),parseInt(start[0]),0],[parseInt(size[1]),parseInt(size[0]),3])
            inputImage=inputImage.resizeBilinear([224,224]).reshape([1,224,224,3])
            result=mask_model.predict(inputImage).dataSync()
            result= Array.from(result)
            ctx.beginPath()
            if (result[1]>result[0]){
              //no mask on
                ctx.strokeStyle="red"
                ctx.fillStyle = "red";
                text = "No Mask: "+(result[1]*100).toPrecision(3).toString()+"%";
            }else{
              //mask on
                ctx.strokeStyle="green"
                ctx.fillStyle = "green";
                text = "Mask: "+(result[0]*100).toPrecision(3).toString()+"%";
            }
              ctx.lineWidth = "4"
            ctx.rect(start[0], start[1],size[0], size[1])
            ctx.stroke()
            ctx.font = "bold 15pt sans-serif";
            ctx.fillText(text,start[0]+5,start[1]+20)
          }     
      }
      //update frame
      requestAnimationFrame(renderPrediction);
      tf.engine().endScope()
    };
  
    const setupPage = async () => {
        await tf.setBackend(state.backend);
        await setupCamera();
        videoRef.current.play();
  
        videoWidth = videoRef.current.videoWidth;
        videoHeight = videoRef.current.videoHeight;
        videoRef.current.width = videoWidth;
        videoRef.current.height = videoHeight;
  
        canvas = document.getElementById('output');
        canvas.width = videoWidth;
        canvas.height = videoHeight;
        ctx = canvas.getContext('2d');
        ctx.fillStyle = "rgba(255, 0, 0, 0.5)"; 
  
        model = await blazeface.load();
        
        mask_model = await tf.loadLayersModel('./static/models/model.json');
  
       renderPrediction();
    };
    setupPage();
  }, [])
  
	
 


  return (
    <div className="App">
      <div className="container p-0" >
		<header >
	      <nav className="navbar navbar-dark bg-dark rounded">
	        <a className="navbar-brand">
	        	<i className="fas fa-camera"></i>
	        	<strong>Masks Detection</strong>
	        </a>
	      </nav>
	    </header>
	</div>
  <div className="container ">
		<div className="row bg-light" style={{height:"480px"}}>
			<video ref={videoRef} id="video" playsinline className="border " style={{margin:"auto",display:"inline-block"}}></video>
			<canvas id="output" className="canvas-output" style={{margin:"auto",position:"relative",top:"-480px",left:"10px"}} />
			<div className="float-right">
	    		<a href="https://www.youtube.com/user/chirpieful">
	    			<img src="../static/img/yicong.jpg" className="rounded" style={{height:"30px",width:"30px",position:"relative",top:"-30px",}} />
	    		</a>
	    	</div>
		</div> 
    </div>
    </div>
  );
}

export default App;
