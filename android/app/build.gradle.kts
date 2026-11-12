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
        versionCode = 2
        versionName = "1.0.2-beta.1"
        // x86_64 for the emulator, arm64-v8a for phones (the CMake file sets each ABI's CPU flags)
        ndk { abiFilters += listOf("x86_64", "arm64-v8a") }
        // androidTest/RefinerBenchTest: the refiner benchmark on a real phone (Firebase Test Lab)
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        externalNativeBuild {
            cmake {
                // 16 KB ELF segment alignment for every native library we build (ggml, llama, the refiner JNI):
                // 4 KB-aligned libraries made Android 16 show PageSizeMismatchDialog over the app (Pixel 10 Pro,
                // Test Lab, 2026-11-08), and Google Play requires 16 KB support for apps targeting Android 15+.
                arguments += listOf("-DLLAMA_DIR=" + project.property("phonon.llamaDir"), "-DCMAKE_BUILD_TYPE=Release",
                    "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON")
                cppFlags += listOf("-O3")
            }
        }
    }
    // Release signing: the keystore stays outside the repository. Pass -Pphonon.releaseStoreFile=PATH and set
    // PHONON_ANDROID_STORE_PASSWORD (key alias "phonon"); without them the release APK is built unsigned.
    val releaseStore = project.findProperty("phonon.releaseStoreFile") as String?
    signingConfigs {
        if (releaseStore != null) {
            create("release") {
                storeFile = file(releaseStore)
                storePassword = System.getenv("PHONON_ANDROID_STORE_PASSWORD")
                keyAlias = "phonon"
                keyPassword = System.getenv("PHONON_ANDROID_STORE_PASSWORD")
            }
        }
    }
    buildTypes {
        debug { isJniDebuggable = false; externalNativeBuild { cmake { arguments += "-DCMAKE_BUILD_TYPE=Release" } } }
        release { if (releaseStore != null) signingConfig = signingConfigs.getByName("release") }
    }
    // -Pphonon.testBuildType=release runs the instrumentation tests (Test Lab) against the release APK.
    testBuildType = (project.findProperty("phonon.testBuildType") as String?) ?: "debug"
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    // sherpa-onnx prebuilt JNI libs (libsherpa-onnx-jni.so, libonnxruntime.so) are copied in by scripts/sync_build.sh
    sourceSets["main"].jniLibs.srcDirs("src/main/jniLibs")
    // extracted to disk, so ggml can dlopen its CPU variant modules by path
    packaging { jniLibs { useLegacyPackaging = true } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    // VoiceIME: the Mac's WebSocket API (MacClient.kt)
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    // TestActivity: "Scan pairing code" reads the Mac's QR code (no Google Play services needed)
    implementation("com.journeyapps:zxing-android-embedded:4.3.0")
    // ...whose scanner screen calls androidx.core (ContextCompat) without depending on it
    implementation("androidx.core:core:1.13.1")
    // LineBreaksTest (JVM): the real org.json, since android.jar's is a stub in local unit tests
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("junit:junit:4.13.2")
}
