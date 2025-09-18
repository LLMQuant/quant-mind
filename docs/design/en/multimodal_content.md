# QuantMind Multimodal Content Processing: Design Document

## 1. Introduction & Motivation (The "Why")

The QuantMind framework is designed to process vast amounts of unstructured financial content into a structured, queryable knowledge base. The **Multimodal Content Processing Layer** extends this capability to handle diverse media types beyond traditional text, enabling comprehensive analysis of financial information across multiple modalities.

Its primary motivations are:

- **Unified Media Processing**: To provide a consistent interface for processing image, audio, and video content alongside traditional text, enabling comprehensive analysis of financial information across all media types.
- **Content Understanding**: To transform raw media files into structured, searchable text representations through advanced parsing and AI-powered understanding techniques.
- **Modular Architecture**: To separate media-specific processing logic from the core knowledge management system, allowing easy extension to new media types without affecting existing functionality.
- **Embedding Generation**: To generate rich, multi-dimensional embeddings from multimodal content, enabling semantic search and retrieval across different media types.
- **Scalable Processing**: To handle large volumes of multimedia content efficiently, with support for batch processing and parallel execution.

## 2. Architecture & Core Concepts (The "How")

The multimodal content processing layer is built on a set of core principles: a unified media content model, specialized knowledge items, and modular parsers with understanding capabilities.

### 2.1. `MediaContent`: The Unified Base Model

The architecture starts with `quantmind.models.media.MediaContent`, a Pydantic model that serves as the foundation for all multimodal content types. This model directly inherits from `BaseModel` for maximum flexibility and provides a consistent interface across all media types.

The model manages four core aspects of multimodal content:

- **Media Metadata**: Basic information about the media file (path, size, duration, resolution)
- **Processing Results**: Parsed text content and AI-generated understanding
- **Processing State**: Flags indicating parsing and understanding completion status
- **Processing Metadata**: Technical details about the processing methods and results

```python
class MediaContent(BaseModel):
    # Core identifiers
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., min_length=1)
    source: Optional[str] = None
    
    # Media-specific fields
    media_type: MediaType = Field(..., description="Type of media content")
    file_path: Optional[str] = Field(None, description="Path to media file")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    resolution: Optional[str] = Field(None, description="Resolution")
    
    # Processing results
    parsed_content: Optional[str] = Field(None, description="Parsed text content")
    understanding: Optional[str] = Field(None, description="AI understanding/summary")
    key_insights: List[str] = Field(default_factory=list, description="Key insights")
    
    # Processing state
    is_parsed: bool = Field(default=False)
    is_understood: bool = Field(default=False)
```

### 2.2. Specialized Knowledge Items

To handle the unique characteristics of different media types, the system provides specialized knowledge item classes that extend `MediaContent`:

#### 2.2.1. `AudioKnowledgeItem`

Represents audio content with specialized fields for speech processing and audio analysis:

```python
class AudioKnowledgeItem(MediaContent):
    # Audio-specific fields
    sample_rate: Optional[int] = Field(None, description="Audio sample rate")
    channels: Optional[int] = Field(None, description="Number of audio channels")
    bitrate: Optional[int] = Field(None, description="Audio bitrate")
    
    # Audio processing results
    transcript: Optional[str] = Field(None, description="Speech-to-text transcript")
    speakers: List[Dict[str, Any]] = Field(default_factory=list, description="Speaker information")
    audio_features: Dict[str, Any] = Field(default_factory=dict, description="Audio features")
    
    # Embeddings
    audio_embedding: Optional[List[float]] = Field(None, description="Audio embedding vector")
    transcript_embedding: Optional[List[float]] = Field(None, description="Transcript embedding vector")
```

#### 2.2.2. `ImageKnowledgeItem`

Represents image content with specialized fields for visual analysis and OCR:

```python
class ImageKnowledgeItem(MediaContent):
    # Image-specific fields
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")
    format: Optional[str] = Field(None, description="Image format")
    
    # Image processing results
    ocr_text: Optional[str] = Field(None, description="OCR extracted text")
    image_description: Optional[str] = Field(None, description="Image description")
    detected_objects: List[Dict[str, Any]] = Field(default_factory=list, description="Detected objects")
    
    # Embeddings
    image_embedding: Optional[List[float]] = Field(None, description="Image embedding vector")
    visual_embedding: Optional[List[float]] = Field(None, description="Visual features embedding vector")
```

#### 2.2.3. `VideoKnowledgeItem`

Represents video content with specialized fields for frame analysis and scene understanding:

```python
class VideoKnowledgeItem(MediaContent):
    # Video-specific fields
    fps: Optional[float] = Field(None, description="Frames per second")
    codec: Optional[str] = Field(None, description="Video codec")
    has_audio: bool = Field(default=False, description="Whether video contains audio")
    
    # Video processing results
    key_frames: List[Dict[str, Any]] = Field(default_factory=list, description="Key frame information")
    video_transcript: Optional[str] = Field(None, description="Video audio transcript")
    scene_analysis: Dict[str, Any] = Field(default_factory=dict, description="Scene analysis results")
    
    # Embeddings
    video_embedding: Optional[List[float]] = Field(None, description="Video embedding vector")
    frame_embeddings: List[List[float]] = Field(default_factory=list, description="Frame embedding vectors")
```

### 2.3. Modular Parser Architecture

The system employs a modular parser architecture where each media type has its dedicated parser implementing a consistent interface:

#### 2.3.1. `MediaParser`: The Abstract Contract

All parsers inherit from `quantmind.parsers.base.MediaParser`, which defines the universal interface for media processing:

```python
class MediaParser(ABC):
    @abstractmethod
    def parse_content(self, media_content: MediaContent) -> MediaContent:
        """Parse media content to extract text information."""
        pass
    
    @abstractmethod
    def understand_content(self, media_content: MediaContent) -> MediaContent:
        """Understand media content and generate insights."""
        pass
    
    def process_content(self, media_content: MediaContent) -> MediaContent:
        """Process media content (parse + understand) in one step."""
        parsed_content = self.parse_content(media_content)
        understood_content = self.understand_content(parsed_content)
        return understood_content
```

#### 2.3.2. Specialized Parsers

Each media type has its dedicated parser with specialized processing capabilities:

- **`ImageParser`**: Handles OCR text extraction, visual analysis, and object detection
- **`AudioParser`**: Manages speech-to-text conversion, speaker identification, and audio feature analysis
- **`VideoParser`**: Processes frame extraction, scene analysis, and audio transcription

### 2.4. Two-Stage Processing Pipeline

The multimodal processing follows a consistent two-stage pipeline:

#### Stage 1: Content Parsing
- **Purpose**: Extract structured information from raw media files
- **Output**: Populated `parsed_content` field with extracted text and metadata
- **Methods**: OCR, speech recognition, frame analysis, object detection

#### Stage 2: Content Understanding
- **Purpose**: Generate AI-powered insights and summaries from parsed content
- **Output**: Populated `understanding` and `key_insights` fields
- **Methods**: LLM analysis, insight extraction, summary generation

### 2.5. Embedding Generation

Each knowledge item type supports multiple embedding types for different use cases:

- **Content Embeddings**: Generated from the combined text content (title + parsed content + understanding)
- **Modality-Specific Embeddings**: Specialized embeddings for each media type (e.g., audio embeddings, visual embeddings)
- **Multi-Modal Embeddings**: Cross-modal embeddings that capture relationships between different media types

## 3. Processing Workflow

### 3.1. Basic Processing Flow

```python
# 1. Create media content
image_item = ImageKnowledgeItem(
    title="Financial Chart Analysis",
    file_path="/path/to/chart.png"
)

# 2. Initialize parser
parser = ImageParser()

# 3. Process content (parse + understand)
processed_item = parser.process_content(image_item)

# 4. Generate embeddings
processed_item.set_image_embedding(image_embedding, "clip-vit-base")
processed_item.set_visual_embedding(visual_embedding, "resnet50")
```

### 3.2. Batch Processing

The system supports efficient batch processing for multiple media items:

```python
# Create multiple media items
media_items = [
    ImageKnowledgeItem(title="Chart 1", file_path="chart1.png"),
    AudioKnowledgeItem(title="Interview 1", file_path="interview1.wav"),
    VideoKnowledgeItem(title="Demo 1", file_path="demo1.mp4")
]

# Process by type
parsers = {
    MediaType.IMAGE: ImageParser(),
    MediaType.AUDIO: AudioParser(),
    MediaType.VIDEO: VideoParser()
}

for item in media_items:
    parser = parsers[item.media_type]
    processed_item = parser.process_content(item)
```

### 3.3. Configuration and Customization

Each parser supports extensive configuration for different processing requirements:

```python
# Image parser configuration
image_config = {
    "ocr_engine": "tesseract",
    "vision_model": "gpt-4-vision",
    "confidence_threshold": 0.8
}

# Audio parser configuration
audio_config = {
    "speech_model": "whisper",
    "language": "zh",
    "sample_rate": 16000,
    "confidence_threshold": 0.8
}

# Video parser configuration
video_config = {
    "frame_extraction_interval": 5,
    "max_frames": 20,
    "audio_extraction": True,
    "vision_model": "gpt-4-vision",
    "speech_model": "whisper"
}
```

## 4. Integration with Storage Layer

The multimodal content processing layer integrates seamlessly with the existing QuantMind storage system:

### 4.1. Storage Compatibility

All `MediaContent` objects can be stored using the existing `BaseStorage` interface:

```python
# Store processed media content
storage = LocalStorage(config)
storage.process_knowledge(processed_image_item)
storage.process_knowledge(processed_audio_item)
storage.process_knowledge(processed_video_item)
```

### 4.2. Embedding Storage

The system leverages the existing embedding storage infrastructure:

```python
# Store embeddings
storage.store_embedding(
    content_id=processed_item.id,
    embedding=processed_item.get_text_for_embedding(),
    model="sentence-transformers/all-MiniLM-L6-v2"
)
```

### 4.3. Index Integration

Multimodal content is automatically indexed and searchable through the existing storage indexing system.

## 5. Conclusion

The QuantMind multimodal content processing layer provides a robust, scalable, and extensible foundation for handling diverse media types in financial knowledge management. By combining a unified content model with specialized processing capabilities and seamless storage integration, it enables comprehensive analysis of financial information across all media modalities. The modular architecture ensures easy extension to new media types while maintaining consistency with the existing QuantMind ecosystem.

The two-stage processing pipeline (parsing + understanding) ensures that all media content is transformed into structured, searchable text representations, enabling powerful semantic search and retrieval capabilities across the entire knowledge base. This design positions QuantMind as a comprehensive solution for multimodal financial knowledge management.
