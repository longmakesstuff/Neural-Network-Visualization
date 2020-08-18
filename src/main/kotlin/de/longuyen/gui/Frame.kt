package de.longuyen.gui

import de.longuyen.ONE_COLOR
import de.longuyen.SIZE
import de.longuyen.ZERO_COLOR
import de.longuyen.neuralnetwork.NeuralNetwork
import de.longuyen.neuralnetwork.activations.LeakyRelu
import de.longuyen.neuralnetwork.activations.Sigmoid
import de.longuyen.neuralnetwork.initializers.XavierInitializer
import de.longuyen.neuralnetwork.losses.CrossEntropy
import de.longuyen.neuralnetwork.metrics.BinaryAccuracy
import de.longuyen.neuralnetwork.optimizers.MomentumGradientDescent
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.awt.*
import java.awt.image.BufferedImage
import javax.swing.JFrame
import javax.swing.JPanel


class Frame(private var xs: Array<IntArray>, private var ys: Array<IntArray>) : JFrame("Feedforward neural network's hypothesis space visualization") {
    private val leftImage = BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_ARGB)
    private val rightImage = BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_ARGB)
    private val rightCanvas = rightImage.graphics
    private val leftCanvas = leftImage.graphics
    private val leftPanel = object : JPanel() {
        override fun paintComponent(g: Graphics) {
            super.paintComponent(g)
            g.drawImage(leftImage, 0, 0, null)
        }
    }
    private val rightPanel = object : JPanel() {
        override fun paintComponent(g: Graphics) {
            super.paintComponent(g)
            g.drawImage(rightImage, 0, 0, null)
        }
    }
    private val layers = intArrayOf(2, 10, 10, 10, 1)
    private val neuralNetwork = NeuralNetwork(
        layers,
        XavierInitializer(),
        LeakyRelu(),
        Sigmoid(),
        CrossEntropy(),
        MomentumGradientDescent(0.1),
        BinaryAccuracy()
    )
    private val xTest: INDArray
    private var xTrain = Nd4j.createFromArray(xs).castTo(DataType.DOUBLE).div(SIZE.toDouble()).transpose()
    private var yTrain = Nd4j.createFromArray(ys).castTo(DataType.DOUBLE).transpose()

    init {
        val xTestArray = mutableListOf<DoubleArray>()
        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {
                xTestArray.add(doubleArrayOf(x.toDouble(), y.toDouble()))
            }
        }
        xTest = Nd4j.createFromArray(xTestArray.toTypedArray()).div(SIZE.toDouble()).transpose()

        setSize(SIZE * 2, SIZE + 25)
        layout = GridLayout(1, 2)
        add(leftPanel)
        add(rightPanel)

        defaultCloseOperation = EXIT_ON_CLOSE
        isVisible = true
    }

    fun run() {
        while (true) {
            neuralNetwork.train(xTrain, yTrain, xTrain, yTrain, 10)

            visualizePrediction()
            visualizeTrainingData()
            visualizeNeuralNetwork()
        }
    }

    private fun visualizeTrainingData() {
        for (i in xs.indices) {
            if (ys[i][0] == 1) {
                rightCanvas.color = ONE_COLOR
            } else {
                rightCanvas.color = ZERO_COLOR
            }
            val x = xs[i][0]
            val y = xs[i][1]
            rightCanvas.fillOval(x - 5, y - 5, 10, 10)
            rightCanvas.color = Color.LIGHT_GRAY
            rightCanvas.drawOval(x - 5, y - 5, 10, 10);
        }
        rightPanel.repaint()
    }

    private fun visualizePrediction() {
        val prediction = neuralNetwork.inference(xTest).reshape(intArrayOf(SIZE, SIZE)).toDoubleMatrix()
        for (y in prediction.indices) {
            for (x in prediction[y].indices) {
                if (prediction[y][x] > 0.5) {
                    val color = Color(100, 178, 255, 255 - ((1.0 - prediction[y][x]) * 255).toInt())
                    rightImage.setRGB(x, y, color.rgb)
                } else {
                    val color = Color(255, 0, 255, 255 - (prediction[y][x] * 255).toInt())
                    rightImage.setRGB(x, y, color.rgb)
                }
            }
        }
        rightPanel.repaint()
    }

    private fun visualizeNeuralNetwork(){
        val offSets = mutableListOf(
            intArrayOf(10, 200),
            intArrayOf(130, 20),
            intArrayOf(250, 20),
            intArrayOf(370, 20),
            intArrayOf(490, 250)
        )
        val margins = intArrayOf(100, 50, 50, 50, 0)
        val radius = 10
        val layerCoordinates = mutableListOf<Array<IntArray>>()

        for(layer in layers.indices){
            val layerCoordinate = mutableListOf<IntArray>()
            for(neuron in 0 until layers[layer]){
                val x = offSets[layer][0]
                val y = offSets[layer][1] + neuron * margins[layer]
                layerCoordinate.add(intArrayOf(x, y))
            }
            layerCoordinates.add(layerCoordinate.toTypedArray())
        }
        for(i in layerCoordinates.indices){
            for(j in layerCoordinates[i].indices){
                val x = layerCoordinates[i][j][0]
                val y = layerCoordinates[i][j][1]
                leftCanvas.color = Color(255, 0, 0, 125)
                leftCanvas.fillOval(x - radius, y - radius, radius * 2, radius * 2)
                leftCanvas.color = Color.LIGHT_GRAY
                leftCanvas.drawOval(x - radius, y - radius, radius * 2, radius * 2)
            }
        }

        val weights = neuralNetwork.toPrimitiveWeights()
        val leftCanvas2D = leftCanvas as Graphics2D
        val oldStroke = leftCanvas2D.stroke
        for(layerIndex in 0 until layerCoordinates.size - 1){
            val currentLayer = layerCoordinates[layerIndex]
            val nextLayer = layerCoordinates[layerIndex + 1]
            for(currentNeuronIndex in currentLayer.indices){
                for(nextNeuronIndex in nextLayer.indices){
                    val currentNeuron = currentLayer[currentNeuronIndex]
                    val nextNeuron = nextLayer[nextNeuronIndex]
                    val weight = weights[layerIndex][nextNeuronIndex][currentNeuronIndex]
                    leftCanvas2D.stroke = BasicStroke(weight.toFloat())
                    leftCanvas2D.color = Color(50, 50, 50, 125)
                    leftCanvas2D.drawLine(currentNeuron[0], currentNeuron[1], nextNeuron[0], nextNeuron[1])
                }
            }
        }
        leftCanvas2D.stroke = oldStroke
        leftPanel.repaint()
    }
}