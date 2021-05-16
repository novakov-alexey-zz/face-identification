Global / onChangedBuildSource := ReloadOnSourceChanges

scalaVersion := "3.0.0-RC3"

libraryDependencies ++= Seq(
  "org.emergent-order" %% "onnx-scala-backends" % "0.13.0",
  "org.bytedeco" % "javacv-platform" % "1.5.5",
  "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.2",
  "io.bullet" % "borer-core_2.13" % "1.7.2"
)

scalacOptions ++= Seq(
  "-deprecation",
  "-encoding",
  "UTF-8",
  "-feature",
  "-unchecked"
)

fork := true
 javaOptions ++= Seq(
   "-Dsun.java2d.opengl=True",
   "-Dsun.java2d.xrender=True"
 )
