package de.longuyen.neuralnetwork.metrics

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class Jaccard : Metric(){
    override fun compute(yTrue: INDArray, yPrediction: INDArray): Double {
        val yTrueArgMax = Nd4j.argMax(yTrue, 0).toDoubleVector()
        val yPredictionArgMax = Nd4j.argMax(yPrediction, 0).toDoubleVector()
        var ab = 0.0
        for(y in yTrueArgMax.indices){
            if(yTrueArgMax[y] == yPredictionArgMax[y]){
                ++ab
            }
        }
        return ab / (yTrueArgMax.size + yPredictionArgMax.size - ab)
    }
}