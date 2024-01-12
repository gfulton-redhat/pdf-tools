package org.adoption;

import org.deeplearning4j.nn.graph.ComputationGraph;

import org.opencv.videoio.VideoCapture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.deeplearning4j.nn.api.Model;

import java.net.MalformedURLException;
import java.net.URI;

import static org.deeplearning4j.nn.modelimport.keras.KerasModelImport.*;

public interface Loader {
     Logger LOGGER = LoggerFactory.getLogger(Loader.class);

    default Model getModel(URI uri) throws Exception {
        try {
            ComputationGraph model = importKerasModelAndWeights(uri.toURL().getFile());
            return model;
        }catch (MalformedURLException e){
            LOGGER.error("Unable to get model using URI: "+uri.toString(), e);
            throw e;
        }
    }

    default VideoCapture getVideoCapture(URI uri) throws Exception {
        try{
            VideoCapture capture = new VideoCapture(uri.toURL().getFile());
            return capture;
        }catch (MalformedURLException e){
            LOGGER.error("Unable to get video using URI: "+uri.toString(), e);
            throw e;
        }
    }

}
