// import $repo.`https://clojars.org/repo`
// import $ivy.`opencv:opencv:4.0.0-0`
import $ivy.`org.bytedeco:javacv-platform:1.5.5`

import org.bytedeco.opencv.opencv_core._
import org.bytedeco.opencv.opencv_objdetect._
import org.bytedeco.opencv.opencv_imgproc._
import org.bytedeco.opencv.opencv_calib3d._

import org.bytedeco.opencv.global.opencv_core._
import org.bytedeco.opencv.global.opencv_imgproc._
import org.bytedeco.opencv.global.opencv_imgcodecs._
import org.bytedeco.opencv.global.opencv_calib3d._
import org.bytedeco.opencv.global.opencv_objdetect._

import org.bytedeco.javacv._
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.indexer._

val classifierPath = "lib/data/haarcascades/haarcascade_frontalface_alt2.xml"
val faceDetector = new CascadeClassifier(classifierPath)

val image = imread("processed/fall-2020/DSCF5370-87.jpg")
val height = image.rows()
val width = image.cols()

val faceDetections = new RectVector()
val grayImage = new Mat(height, width, CV_8UC1)
cvtColor(image, grayImage, CV_BGR2GRAY)
faceDetector.detectMultiScale(image, faceDetections)

val total = faceDetections.size.toInt
for (i <- 0 until total) {
  val rect = faceDetections.get(i)
  rectangle(
    image,
    new Point(rect.x, rect.y),
    new Point(rect.x + rect.width, rect.y + rect.height),
    AbstractScalar.GREEN
  )
}

// Save the visualized detection.
val filename = "faceDetection.png"
println(s"Writing $filename")
imwrite(filename, image)
