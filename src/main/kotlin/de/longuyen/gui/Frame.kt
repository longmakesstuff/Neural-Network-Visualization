package de.longuyen.gui

import de.longuyen.*
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
import java.io.File
import javax.imageio.ImageIO
import javax.swing.JFrame
import javax.swing.JPanel
import kotlin.math.abs
import kotlin.math.min


fun joinBufferedImage(img1: BufferedImage, img2: BufferedImage): BufferedImage {
    val width = img1.width + img2.width
    val height = img1.height
    val newImage = BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
    val g2 = newImage.createGraphics()
    val oldColor = g2.color
    g2.paint = Color.WHITE
    g2.fillRect(0, 0, width, height)
    g2.color = oldColor
    g2.drawImage(img1, null, 0, 0)
    g2.drawImage(img2, null, img1.width, 0)
    g2.dispose()
    return newImage
}

fun gradientColor(x: Float, minX: Double = 0.0, maxX: Double = 2.0, from: Color = ZEROS_COLOR, to: Color = ONES_COLOR): Color {
    val range = maxX - minX
    val q = min(x.toDouble(), maxX)
    val p = (q - minX) / range

    val red = (from.red * p + to.red * (1 - p)).toInt()
    val green = (from.green * p + to.green * (1 - p)).toInt()
    val blue = (from.blue * p + to.blue * (1 - p)).toInt()

    return Color(red, green, blue, 125)
}


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
    private val neuralNetwork = NeuralNetwork(
        LAYERS,
        XavierInitializer(),
        LeakyRelu(),
        Sigmoid(),
        CrossEntropy(),
        MomentumGradientDescent(LEARNING_RATE),
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

        setSize(SIZE * 2, (SIZE + (SIZE * 0.125)).toInt())
        layout = GridLayout(1, 2)
        add(leftPanel)
        add(rightPanel)

        setLocationRelativeTo(null)
        defaultCloseOperation = EXIT_ON_CLOSE
        isVisible = true
    }

    fun run() {
        var i = 0
        while (true) {
            neuralNetwork.train(xTrain, yTrain, xTrain, yTrain, 1)

            visualizePrediction()
            visualizeTrainingData()
            visualizeNeuralNetwork()
            if(i % SAVE_FREQUENCE == 0) {
                val image = joinBufferedImage(leftImage, rightImage)
                ImageIO.write(image, "png", File("target/${i}.png"))
            }
            i++
        }
    }

    private fun visualizeTrainingData() {
        for (i in xs.indices) {
            if (ys[i][0] == 1) {
                rightCanvas.color = ZEROS_COLOR
            } else {
                rightCanvas.color = ONES_COLOR
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
        val startX = SIZE / LAYERS.size / 2
        val xOffset = SIZE / LAYERS.size
        val offSets = mutableListOf<IntArray>()
        for(i in LAYERS.indices){
            val yOffset = SIZE / LAYERS[i] / 2
            offSets.add(intArrayOf(startX + i * xOffset, yOffset))
        }
        val margins = mutableListOf<Int>()
        for(element in LAYERS){
            margins.add(SIZE / element)
        }
        val radius = 10
        val layerCoordinates = mutableListOf<Array<IntArray>>()

        for(layer in LAYERS.indices){
            val layerCoordinate = mutableListOf<IntArray>()
            for(neuron in 0 until LAYERS[layer]){
                val x = offSets[layer][0]
                val y = offSets[layer][1] + neuron * margins[layer]
                layerCoordinate.add(intArrayOf(x, y))
            }
            layerCoordinates.add(layerCoordinate.toTypedArray())
        }
        val minMax = neuralNetwork.minMax()
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
                    leftCanvas2D.stroke = BasicStroke(abs(weight.toFloat()) * 3)
                    leftCanvas2D.color = gradientColor(weight.toFloat(), minX = minMax.first, maxX = minMax.second)
                    leftCanvas2D.drawLine(currentNeuron[0], currentNeuron[1], nextNeuron[0], nextNeuron[1])
                }
            }
        }
        leftCanvas2D.stroke = oldStroke
        leftPanel.repaint()

        val forward = neuralNetwork.forward(xTest)
        forward["Z0"] = xTest
        for(i in layerCoordinates.indices){
            val layerOutput = forward["Z$i"]!!
            val minLayerOutput = layerOutput.min().element() as Double
            val maxLayerOutput = layerOutput.max().element() as Double
            val layerOutputVector = layerOutput.mean(1).toDoubleVector()
            for(j in layerCoordinates[i].indices){
                val x = layerCoordinates[i][j][0]
                val y = layerCoordinates[i][j][1]

                leftCanvas2D.color = gradientColor(layerOutputVector[j].toFloat(), minLayerOutput, maxLayerOutput)
                leftCanvas2D.fillOval(x - radius, y - radius, radius * 2, radius * 2)
                leftCanvas2D.drawOval(x - radius, y - radius, radius * 2, radius * 2)
            }
        }
        val titleFont = Font("SansSerif", Font.BOLD, 10)
        leftCanvas2D.font = titleFont
        val x1 = layerCoordinates.first()[0][0]
        val y1 = layerCoordinates.first()[0][1]
        leftCanvas2D.color = Color.BLACK
        leftCanvas2D.drawString("X", x1 - 20, y1 + 3)
        val x2 = layerCoordinates.first()[1][0]
        val y2 = layerCoordinates.first()[1][1]
        leftCanvas2D.color = Color.BLACK
        leftCanvas2D.drawString("Y", x2 - 20, y2 + 3)
        val x3 = layerCoordinates.last()[0][0]
        val y3 = layerCoordinates.last()[0][1]
        leftCanvas2D.color = Color.BLACK
        leftCanvas2D.drawString("Out", x3 + 12, y3 + 3)

        leftPanel.repaint()
    }
}