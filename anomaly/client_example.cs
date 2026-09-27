// C# 에서 검사 API 를 호출하는 예제 (.NET 6 이상, 추가 패키지 없음)
// 사용: InspectionClient.InspectAsync("C:\\images\\part.png") 호출 -> 판정/점수/불량 좌표
//
// 주의: 이 파일은 참고용 예제이며 이 PC 에는 .NET 이 없어 컴파일 검증은 하지 않았습니다.
//       라인 PC 의 개발 환경에서 빌드 후 사용하세요.

using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

public class InspectionResult
{
    [JsonPropertyName("result")] public string Result { get; set; }        // "OK" 또는 "NG"
    [JsonPropertyName("score")] public double Score { get; set; }          // 0~100
    [JsonPropertyName("threshold")] public double Threshold { get; set; }  // 0~100
    [JsonPropertyName("boxes")] public int[][] Boxes { get; set; }         // [x, y, w, h]
    [JsonPropertyName("elapsed_ms")] public double ElapsedMs { get; set; }
    [JsonPropertyName("model")] public string Model { get; set; }
    [JsonPropertyName("heatmap_png_base64")] public string HeatmapPngBase64 { get; set; } // return_heatmap=true 일 때만
}

public static class InspectionClient
{
    // 연결을 재사용해야 요청마다 접속 비용이 생기지 않는다 (TACT 에 중요)
    private static readonly HttpClient Http = new HttpClient
    {
        BaseAddress = new Uri("http://127.0.0.1:8000/"),
        Timeout = TimeSpan.FromSeconds(30),
    };

    public static async Task<InspectionResult> InspectAsync(
        string imagePath, string model = null, double? threshold = null, bool returnHeatmap = false)
    {
        byte[] bytes = await System.IO.File.ReadAllBytesAsync(imagePath);
        return await InspectAsync(bytes, System.IO.Path.GetFileName(imagePath), model, threshold, returnHeatmap);
    }

    // 카메라에서 바로 받은 바이트 배열(PNG/JPG 인코딩된 것)을 보낼 때
    public static async Task<InspectionResult> InspectAsync(
        byte[] imageBytes, string fileName, string model = null, double? threshold = null, bool returnHeatmap = false)
    {
        using var form = new MultipartFormDataContent();
        var fileContent = new ByteArrayContent(imageBytes);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
        form.Add(fileContent, "file", fileName);
        if (model != null) form.Add(new StringContent(model), "model");
        if (threshold.HasValue) form.Add(new StringContent(threshold.Value.ToString("0.###")), "threshold");
        form.Add(new StringContent(returnHeatmap ? "true" : "false"), "return_heatmap");

        using var resp = await Http.PostAsync("inspect", form);
        resp.EnsureSuccessStatusCode();
        string json = await resp.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<InspectionResult>(json);
    }
}

// ---- 사용 예 ----
// var r = await InspectionClient.InspectAsync(@"C:\images\part_001.png");
// Console.WriteLine($"{r.Result}  score={r.Score}  boxes={r.Boxes.Length}  {r.ElapsedMs}ms");
// if (r.Result == "NG") { /* NG 배출 신호 */ }
//
// 히트맵이 필요하면 returnHeatmap: true 로 호출한 뒤
// byte[] png = Convert.FromBase64String(r.HeatmapPngBase64);  // 파일 저장 또는 화면 표시
