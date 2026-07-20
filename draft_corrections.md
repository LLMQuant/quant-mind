# Draft Corrections: Verified Memory Repo Shortlist

## 1) Dogrulama olcutu

Bu calisma "en iyi" iddiasi degil, "dogrulanmis adaylar" uretir. Bir repo shortlist'e girmek icin asagidaki kosullari birlikte saglamalidir:

1. Son 90 gunde aktif commit hareketi var.
2. Gercek bir memory architecture sunuyor (tier, retrieval, persistence, governance gibi somut katmanlar).
3. Agentic AI veya autonomous agent akislarina dogrudan temas ediyor.
4. Acik dokumantasyon ve calisan ornek kullanim veriyor.
5. Yildiz/fork/adoption sinyali ile topluluk ilgisi goruluyor.

Degerlendirme notu:
- Stars tek basina karar kriteri degildir.
- Mimari netlik + bakim aktivitesi + entegrasyon kaniti birincil agirliktir.

## 2) Haric tutma kurallari

Asagidaki repo tipleri shortlist disinda tutulur:

1. Sadece repo adi veya README icinde "memory" geciyor, fakat gercek memory altyapisi yok.
2. Genel framework sunuyor, ancak memory disiplini (state lifecycle, retrieval policy, persistence strategy) icermiyor.
3. Eski, pasif, bakim disi veya son donemde anlamli gelistirme gostermeyen projeler.
4. Mimariyi dogrulayan kanitlar eksik (zayif docs, belirsiz API, calismayan ornekler).

## 3) QuantMind icin cikarim

### Copilot Memory benzeri patternler

- Citation-backed facts
- Validation before use
- Stale item cleanup
- Repo-scoped memory

### QuantMind'e dogrudan uyarlama

- `RawFallbackNode`: Yapilandirilamayan veya dusuk guvenli ciktilari kaybetmeden ham iz olarak saklama.
- `QualityGate`: Durable katmana commit oncesi sema, kaynak ve tutarlilik denetimi.
- `Hybrid memory tiers`: Working, episodic ve durable katmanlari amaca gore ayirma.
- `Graph only for distilled facts`: Graph katmanina yalnizca rafine edilmis, kaynaklanmis ve tekrar dogrulanmis bilgi yazma.

### L3 commit icin zorunlu kosullar

`confidence >= 0.85` tek basina yeterli degildir. Asagidaki kosullar birlikte zorunludur:

1. Provenance mevcut ve dogrulanabilir.
2. Schema validity basarili.
3. Dedup ve contradiction kontrolu basarili.

## Sonraki adim (uygulama hazirligi)

Bu taslak sonraki iterasyonda su ciktiya donusturulmelidir:

- Dogrulanmis 21 repo shortlist
- Her repo icin memory yaklasimi siniflandirmasi
- QuantMind uyarlanabilirlik puani
- Risk/avantaj analizi
- Kopyalanacak mimari pattern oneri matrisi
