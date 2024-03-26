pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
        maven("https://jitpack.io") // Dodanie JitPack jako repozytorium pluginów
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven("https://jitpack.io") // Dodanie JitPack jako repozytorium zależności
    }
}


rootProject.name = "polarGraph"
include(":app")
 