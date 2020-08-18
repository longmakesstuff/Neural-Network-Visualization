package de.longuyen

import de.longuyen.data.FourQuartersDataGenerator
import de.longuyen.gui.Frame

fun main(){
    val data = FourQuartersDataGenerator(DATA_POINTS).generate()
    val frame = Frame(data.first, data.second)
    frame.run()
}