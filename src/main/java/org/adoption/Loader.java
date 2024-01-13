package org.adoption;

import org.deeplearning4j.nn.graph.ComputationGraph;

import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
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

    /**
     * Get the total number of frames and calculate the frame divisor, rounded to the closest int.
     * @return totalNumberOfFrames
     */
    default int getTotalNumberOfFrames(VideoCapture videoCapture){
        int totalNumberOfFrames = (int) videoCapture.get(Videoio.CAP_PROP_FRAME_COUNT);
        return totalNumberOfFrames;
    }

    /**
     * Get calculate the frame divisor and rounded to the closest int.
     * @return frameDivisor
     */
    default int getFrameDivisor(VideoCapture videoCapture, double period) {
        int frameDivisor = (int) Math.round(period * videoCapture.get(Videoio.CAP_PROP_FPS));
        return frameDivisor;
    }
}
