import org.bytedeco.opencv.opencv_core.{Rect, Mat, Size}
import org.bytedeco.opencv.global.opencv_imgproc.resize
import org.bytedeco.opencv.global.opencv_imgcodecs.{imread, imwrite}

import java.nio.file.{Files, Paths}

def createIfNotExists(path: String) =
  if !Files.exists(Paths.get(path)) then 
    Files.createDirectory(Paths.get(path))

@main
def crop() =
  val datasetDir = "dataset-people"
  createIfNotExists(datasetDir)

  val dirs = Paths.get("raw_photos").toFile.listFiles.filter(f => !f.getName.startsWith("."))

  for dir <- dirs do
    val label = dir.getName
    println(s"Extraction images for $label label")

    createIfNotExists(Paths.get(datasetDir, label).toString)
    val images = dir.listFiles.filter(_.toString.endsWith(".jpg"))

    for file <- images do
      println(s"Reading file: $file")
      val image = imread(file.toString)
      val faces = detectFaces(image)  

      for (face, i) <- faces.get.zipWithIndex do        
        val crop = Rect(face.x, face.y, face.width, face.height)
        val cropped = Mat(image, crop)
        resize(cropped, cropped, Size(ImageHeight, ImageWidth))  
        val filename = Paths.get(datasetDir, label, s"$i-${file.getName}").toString
        println(s"Writing $filename")
        imwrite(filename, cropped)