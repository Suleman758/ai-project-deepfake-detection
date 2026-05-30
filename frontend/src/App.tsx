import { useState, useRef, type FormEvent } from "react"
import { ShaderAnimation } from "@/components/ui/shader-animation"
import { Upload, AlertTriangle, CheckCircle, X, Clock, Video, Camera, Loader2 } from "lucide-react"

interface FrameResult {
  frame: number
  face_detected: boolean
  probability?: number
  prediction?: string
}

interface ApiResult {
  label: string
  confidence: number
  total_frames: number
  duration: number
  fps: number
  faces_detected: number
  frames_checked: number
  frame_results: FrameResult[]
  error: string | null
}

export default function App() {
  const [result, setResult] = useState<ApiResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const form = e.currentTarget
    const fileInput = form.elements.nativeElement?.querySelector('input[type="file"]') as HTMLInputElement
    const file = fileInput?.files?.[0]
    if (!file) return

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append("video", file)

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `Server error: ${res.status}`)
      }
      const data: ApiResult = await res.json()
      if (data.error) {
        setError(data.error)
      } else {
        setResult(data)
      }
    } catch (err: any) {
      setError(err.message || "Failed to connect to server")
    } finally {
      setLoading(false)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setFileName(file.name)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith("video/")) {
      if (fileInputRef.current) {
        const dt = new DataTransfer()
        dt.items.add(file)
        fileInputRef.current.files = dt.files
        setFileName(file.name)
      }
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => setDragOver(false)

  const confidencePct = result ? (result.confidence * 100).toFixed(2) : "0"
  const isFake = result?.label === "FAKE"

  return (
    <div className="relative w-full min-h-screen bg-black overflow-hidden">
      {/* Full-screen shader background */}
      <div className="fixed inset-0 z-0 opacity-70">
        <ShaderAnimation />
      </div>

      {/* Overlay gradient */}
      <div className="fixed inset-0 z-[1] bg-gradient-to-b from-black/70 via-black/40 to-black/80" />

      {/* Content */}
      <div className="relative z-10 flex flex-col items-center min-h-screen px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8 mt-4">

          <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight">
            Deepfake Detector
          </h1>
          <p className="text-white/50 text-sm mt-2 max-w-md mx-auto">
            Upload a video to check if it's real or AI-generated using ResNet50 + XGBoost
          </p>
        </div>

        {/* Upload form */}
        <form
          onSubmit={handleSubmit}
          className="w-full max-w-xl"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <div
            className={`
              relative rounded-2xl border-2 border-dashed p-8 text-center transition-all duration-200
              ${dragOver
                ? "border-cyan-400 bg-cyan-400/10"
                : "border-white/20 bg-white/5 hover:border-white/40 hover:bg-white/10"
              }
              backdrop-blur-sm
            `}
          >
            <input
              ref={fileInputRef}
              type="file"
              name="video"
              accept=".mp4,.avi,.mov,.mkv,.webm,.flv"
              onChange={handleFileChange}
              className="hidden"
              id="video-input"
            />
            <label htmlFor="video-input" className="cursor-pointer block">
              <Upload className="w-10 h-10 mx-auto mb-3 text-white/40" />
              <p className="text-white/70 text-sm font-medium">
                {fileName || "Drop a video or click to browse"}
              </p>
              <p className="text-white/30 text-xs mt-1">
                MP4, AVI, MOV, MKV &bull; Max 500 MB
              </p>
            </label>
          </div>

          <button
            type="submit"
            disabled={loading || !fileName}
            className="
              mt-4 w-full py-3 px-6 rounded-xl font-semibold text-sm
              transition-all duration-200 flex items-center justify-center gap-2
              bg-cyan-500 hover:bg-cyan-400 text-white
              disabled:opacity-40 disabled:cursor-not-allowed
            "
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Video className="w-4 h-4" />
                Analyze Video
              </>
            )}
          </button>
        </form>

        {/* Error */}
        {error && (
          <div className="mt-6 w-full max-w-xl rounded-xl bg-red-500/20 border border-red-500/30 backdrop-blur-sm p-4 flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <p className="text-red-200 text-sm">{error}</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="mt-6 w-full max-w-xl space-y-4">
            {/* Verdict card */}
            <div
              className={`
                rounded-2xl border p-6 backdrop-blur-sm
                ${isFake
                  ? "bg-red-500/10 border-red-500/30"
                  : "bg-emerald-500/10 border-emerald-500/30"
                }
              `}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white/50 text-xs uppercase tracking-widest">Verdict</p>
                  <p className={`text-3xl font-bold mt-1 ${isFake ? "text-red-400" : "text-emerald-400"}`}>
                    {isFake ? "FAKE" : "REAL"}
                  </p>
                  <p className={`text-lg font-semibold mt-1 ${isFake ? "text-red-300" : "text-emerald-300"}`}>
                    {confidencePct}% confidence
                  </p>
                </div>
                <div className={`w-16 h-16 rounded-2xl flex items-center justify-center ${isFake ? "bg-red-500/20" : "bg-emerald-500/20"}`}>
                  {isFake
                    ? <X className="w-8 h-8 text-red-400" />
                    : <CheckCircle className="w-8 h-8 text-emerald-400" />
                  }
                </div>
              </div>
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { icon: Clock, label: "Duration", value: `${result.duration}s` },
                { icon: Camera, label: "Frames", value: `${result.total_frames}` },
                { icon: Video, label: "FPS", value: `${result.fps}` },
                { icon: Camera, label: "Faces", value: `${result.faces_detected} / ${result.frames_checked}` },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="rounded-xl bg-white/5 border border-white/10 backdrop-blur-sm p-3 text-center"
                >
                  <stat.icon className="w-4 h-4 mx-auto mb-1 text-white/30" />
                  <p className="text-white/40 text-xs">{stat.label}</p>
                  <p className="text-white font-semibold text-sm">{stat.value}</p>
                </div>
              ))}
            </div>

            {/* Frame results table */}
            {result.frame_results && result.frame_results.length > 0 && (
              <div className="rounded-2xl bg-white/5 border border-white/10 backdrop-blur-sm overflow-hidden">
                <div className="px-4 py-3 border-b border-white/10">
                  <p className="text-white/60 text-xs uppercase tracking-widest font-medium">Per-Frame Analysis</p>
                </div>
                <div className="max-h-64 overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-black/80">
                      <tr className="border-b border-white/10">
                        <th className="text-left px-4 py-2 text-white/40 font-medium text-xs">Frame</th>
                        <th className="text-left px-4 py-2 text-white/40 font-medium text-xs">Face</th>
                        <th className="text-left px-4 py-2 text-white/40 font-medium text-xs">Prediction</th>
                        <th className="text-right px-4 py-2 text-white/40 font-medium text-xs">Fake %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.frame_results.map((fr) => (
                        <tr key={fr.frame} className="border-b border-white/5 hover:bg-white/5">
                          <td className="px-4 py-2 text-white/60">#{fr.frame}</td>
                          <td className="px-4 py-2">
                            {fr.face_detected ? (
                              <span className="inline-flex items-center gap-1 text-emerald-400 text-xs">
                                <CheckCircle className="w-3 h-3" /> Yes
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 text-white/30 text-xs">
                                <X className="w-3 h-3" /> No
                              </span>
                            )}
                          </td>
                          <td className="px-4 py-2">
                            {fr.face_detected ? (
                              <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
                                fr.prediction === "FAKE"
                                  ? "bg-red-500/20 text-red-300"
                                  : "bg-emerald-500/20 text-emerald-300"
                              }`}>
                                {fr.prediction}
                              </span>
                            ) : (
                              <span className="text-white/20 text-xs">&mdash;</span>
                            )}
                          </td>
                          <td className="px-4 py-2 text-right">
                            {fr.face_detected ? (
                              <span className="text-white/60 text-xs">
                                {(fr.probability! * 100).toFixed(1)}%
                              </span>
                            ) : (
                              <span className="text-white/20 text-xs">&mdash;</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Reset */}
            <button
              onClick={() => { setResult(null); setFileName(null); }}
              className="w-full py-2 text-white/40 hover:text-white/70 text-xs transition-colors"
            >
              Upload another video
            </button>
          </div>
        )}

        {/* Footer */}
        <div className="mt-auto pt-8 pb-4 text-center">
          <p className="text-white/20 text-xs">
            ResNet50 + XGBoost &bull; FaceForensics++ dataset
          </p>
        </div>
      </div>
    </div>
  )
}
