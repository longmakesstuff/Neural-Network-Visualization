package de.longuyen

import de.longuyen.data.TwoHalvesDataGenerator
import java.awt.Color

const val SIZE = 400
const val RADIUS = 100
const val DATA_POINTS = 500
val LAYERS = intArrayOf(2, 10, 5, 1)
const val SAVE_FREQUENCE = 25
const val LEARNING_RATE = 0.075
val DATA_GENERATOR = TwoHalvesDataGenerator(DATA_POINTS)
val ZEROS_COLOR =  Color(100, 178, 255, 255)
val ONES_COLOR = Color(255, 0, 255, 255)