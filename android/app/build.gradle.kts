plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ai.phonon.keyboard"
    compileSdk = 35
    ndkVersion = "27.2.12479018"
    defaultConfig {
        applicationId = "ai.phonon.keyboard"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        // x86_64 for the emulator, arm64-v8a for phones (the CMake file sets each ABI's CPU flags)
        ndk { abiFilters += listOf("x86_64", "arm64-v8a") }
        // androidTest/RefinerBenchTest: the refiner benchmark on a real phone (Firebase Test Lab)
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        externalNativeBuild {
            cmake {
                arguments += listOf("-DLLAMA_DIR=" + project.property("phonon.llamaDir"), "-DCMAKE_BUILD_TYPE=Release")
                cppFlags += listOf("-O3")
            }
        }
    }
    buildTypes {
        debug { isJniDebuggable = false; externalNativeBuild { cmake { arguments += "-DCMAKE_BUILD_TYPE=Release" } } }
    }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    // sherpa-onnx prebuilt JNI libs (libsherpa-onnx-jni.so, libonnxruntime.so) are copied in by scripts/sync_build.sh
    sourceSets["main"].jniLibs.srcDirs("src/main/jniLibs")
    // extracted to disk, so ggml can dlopen its CPU variant modules by path
    packaging { jniLibs { useLegacyPackaging = true } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    // LineBreaksTest (JVM): the real org.json, since android.jar's is a stub in local unit tests
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("junit:junit:4.13.2")
}
