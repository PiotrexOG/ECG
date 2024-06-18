package com.example.senderemulator

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import java.io.BufferedReader
import java.io.DataOutputStream
import java.io.IOException
import java.io.InputStreamReader
import java.net.Socket
import java.nio.ByteBuffer
import java.util.Timer
import java.util.TimerTask
import java.util.concurrent.Executors

private var socket: Socket? = null
private var outputStream: DataOutputStream? = null

data class PolarEcgDataSample(
    val timeStamp: Long,
    val voltage: Int
)

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "ECGActivity"
        private const val SAMPLE_SIZE = 73
    }

    private val samplesList = mutableListOf<PolarEcgDataSample>()
    private var currentIndex = 0
    private val timer = Timer()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Load samples from CSV file
        loadEcgDataFromCsv("ecg_data1.csv")

        scheduleNextPacket(0)
    }

    private fun scheduleNextPacket(initialDelay: Long) {
        timer.schedule(object : TimerTask() {
            override fun run() {
                if (currentIndex + SAMPLE_SIZE <= samplesList.size) {
                    val samples = samplesList.subList(currentIndex, currentIndex + SAMPLE_SIZE)
                    sendDataToServer(samples)
                    Log.d(TAG, samples[0].voltage.toString())
                    Log.d(TAG, samples[0].timeStamp.toString())

                    // Calculate delay based on timestamps in nanoseconds, convert to milliseconds
                    val delayNs = samples.last().timeStamp - samples.first().timeStamp
                    val delayMs = delayNs / 1_000_000

                    currentIndex += SAMPLE_SIZE
                    scheduleNextPacket(delayMs)
                }
            }
        }, initialDelay)
    }

    private fun openSocketIfNeeded() {
        if (socket == null || outputStream == null || socket!!.isClosed) {
            try {
                socket = Socket("192.168.1.132", 12345) // Server IP address
                outputStream = DataOutputStream(socket!!.getOutputStream())
            } catch (e: IOException) {
                Log.e(TAG, "Error opening socket: $e")
            }
        }
    }

    private val executor = Executors.newSingleThreadExecutor()

    private fun sendDataToServer(samples: List<PolarEcgDataSample>) {
        executor.submit {
            try {
                openSocketIfNeeded()

                // Calculate the buffer size: 4 bytes for float (voltage) + 8 bytes for long (timestamp)
                val buffer = ByteBuffer.allocate(samples.size * (4 + 8))
                for (obj in samples) {
                    buffer.putFloat(obj.voltage.toFloat())
                    buffer.putLong(obj.timeStamp)
                }
                val bytes = buffer.array()

                outputStream?.write(bytes)
                outputStream?.flush()
            } catch (e: IOException) {
                Log.e(TAG, "Error sending data: $e")
            }
        }
    }

    private fun loadEcgDataFromCsv(fileName: String) {
        try {
            val inputStream = assets.open(fileName)
            val reader = BufferedReader(InputStreamReader(inputStream))
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                val parts = line!!.split(",")
                if (parts.size == 2) {
                    val timeStamp = parts[0].toLongOrNull()
                    val voltage = parts[1].toIntOrNull()
                    if (timeStamp != null && voltage != null) {
                        samplesList.add(PolarEcgDataSample(timeStamp, voltage))
                    }
                }
            }
            reader.close()
        } catch (e: IOException) {
            Log.e(TAG, "Error reading CSV file: $e")
        }
    }
}
