import org.bytedeco.opencv.opencv_core.{Mat, Size, Rect, Point, Scalar}
import org.bytedeco.opencv.global.opencv_imgproc.*
import org.bytedeco.opencv.opencv_videoio.VideoCapture
import org.bytedeco.javacv.{CanvasFrame, OpenCVFrameConverter}

import javax.swing.WindowConstants

def createCavasFrame = 
  val frame = CanvasFrame("Detected Faces")
  frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
  frame.setCanvasSize(1280, 720)
  frame

def calcLabel(face: Array[Float], features: Features, threshold: Int = 100) = 
  features.foldLeft("?", Float.MaxValue){ 
    case ((label, min), (l, f)) => 
      val d = distance(face, f)
      if d < threshold && d < min then (l, d)
      else (label, min)
  }._1  

def drawLabel(label: String, frame: Mat, topLeft: Point) =
  val x = math.max(topLeft.x - 10, 0)
  val y = math.max(topLeft.y - 10, 0)
  val font = FONT_HERSHEY_SIMPLEX
  val thickness = 2
  val fontScale = 1.0
  val baseline = new Array[Int](2)
  val size = getTextSize(label, font, fontScale, thickness, baseline)
  val rectColor = Scalar(255, 0, 0, 0)
  rectangle(
    frame,
    Point(x, y - size.height() - thickness),
    Point(x + size.width() - thickness, y + 10),
    rectColor,
    CV_FILLED,
    LINE_8,
    0)
  val fontColor = Scalar(0, 255, 0, 0)
  putText(frame, label, Point(x, y), font, fontScale, fontColor, thickness, CV_FILLED, false)

def toModelInput(crop: Rect, frame: Mat) =
  val cropped = Mat(frame, crop)
  resize(cropped, cropped, Size(ImageHeight, ImageWidth))
  toArray(scale(cropped))

def drawRectangle(face: Rect, frame: Mat) =
  rectangle(frame,
    Point(face.x, face.y),
    Point(face.x + face.width, face.y + face.height),
    Scalar(0, 255, 0, 1)
  )

@main
def demo() =
  val capture = VideoCapture(0)
  val canvasFrame = createCavasFrame  
  val frame = Mat()
  val converter = OpenCVFrameConverter.ToMat()
  val model = getModel()
  val features = loadFeatures

  try
    while capture.read(frame) do
      val faces = detectFaces(frame)
      
      for face <- faces.get yield
        drawRectangle(face, frame)
        val crop = Rect(face.x, face.y, face.width, face.height)
        val image = toModelInput(crop, frame)
        val faceFeatures = predict(image, model).data
        val label = calcLabel(faceFeatures, features)
        drawLabel(label, frame, crop.tl)

      canvasFrame.showImage(converter.convert(frame))                              
  finally
    capture.release
    canvasFrame.dispose