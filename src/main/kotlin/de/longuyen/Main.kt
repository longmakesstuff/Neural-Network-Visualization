package de.longuyen

import de.longuyen.data.TwoHalvesDataGenerator
import de.longuyen.gui.Frame

fun main(){
    val data = TwoHalvesDataGenerator(250).generate()
    val frame = Frame(data.first, data.second)
    frame.run()
}