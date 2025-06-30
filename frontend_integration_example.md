# Frontend Integration for Markdown Pipeline

## Update API Service

Replace the complex client-side PDF generation with simple API calls:

```typescript
// src/services/api.ts - New method
async generateMarkdownDocument(request: GenerationRequest): Promise<MarkdownDocumentResult> {
  try {
    const response = await fetch('/api/generation/documents/generate-markdown', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Document generation failed:', error);
    throw error;
  }
}

// Types for the new API
interface MarkdownDocumentResult {
  success: boolean;
  document_id: string;
  markdown_content: string;  // For display
  downloads: {
    markdown: DownloadInfo;
    html: DownloadInfo;
    pdf: DownloadInfo;
    docx: DownloadInfo;
  };
  metadata: DocumentMetadata;
}

interface DownloadInfo {
  available: boolean;
  download_url?: string;
  file_size: number;
}
```

## Update TeacherGenerationPage

Replace the complex rendering with simple markdown display + download buttons:

```tsx
// src/pages/TeacherGenerationPage.tsx - Updated component
const TeacherGenerationPage: React.FC = () => {
  const [markdownResult, setMarkdownResult] = useState<MarkdownDocumentResult | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerate = async (request: GenerationRequest) => {
    setIsGenerating(true);
    try {
      // Use new markdown API instead of complex JSON generation
      const result = await apiService.generateMarkdownDocument(request);
      setMarkdownResult(result);
      
      // Show success message
      toast.success('Document generated successfully!');
      
    } catch (error) {
      toast.error('Generation failed: ' + error.message);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1>Generate Educational Materials</h1>
      
      {/* Generation Form */}
      <GenerationForm onGenerate={handleGenerate} isLoading={isGenerating} />
      
      {/* Results Display */}
      {markdownResult?.success && (
        <div className="mt-8 space-y-6">
          
          {/* Document Preview - Simple markdown display */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold mb-4">Generated Document</h2>
            <div className="prose max-w-none">
              <ReactMarkdown>{markdownResult.markdown_content}</ReactMarkdown>
            </div>
          </div>
          
          {/* Download Section - Clean and simple */}
          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Download Formats</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              
              {Object.entries(markdownResult.downloads).map(([format, info]) => (
                <DownloadButton
                  key={format}
                  format={format}
                  available={info.available}
                  downloadUrl={info.download_url}
                  fileSize={info.file_size}
                />
              ))}
              
            </div>
          </div>
          
        </div>
      )}
    </div>
  );
};

// Simple download button component
const DownloadButton: React.FC<{
  format: string;
  available: boolean;
  downloadUrl?: string;
  fileSize: number;
}> = ({ format, available, downloadUrl, fileSize }) => {
  
  const handleDownload = () => {
    if (downloadUrl) {
      // Direct download from R2 - no client-side processing!
      window.open(downloadUrl, '_blank');
    }
  };

  return (
    <button
      onClick={handleDownload}
      disabled={!available}
      className={`p-4 rounded-lg border-2 transition-colors ${
        available
          ? 'border-blue-500 bg-blue-50 hover:bg-blue-100 text-blue-700'
          : 'border-gray-300 bg-gray-100 text-gray-400 cursor-not-allowed'
      }`}
    >
      <div className="text-center">
        <div className="font-semibold uppercase">{format}</div>
        <div className="text-sm mt-1">
          {available ? `${(fileSize / 1024).toFixed(1)} KB` : 'Not available'}
        </div>
      </div>
    </button>
  );
};
```

## Remove Old Complex Components

Delete or simplify these components that are no longer needed:

```typescript
// ❌ Remove these complex components:
// - DocumentRenderer (replace with ReactMarkdown)
// - DownloadManager (replace with simple download buttons) 
// - Complex content structure parsing
// - Client-side PDF generation (jsPDF)
// - formatContentAsText functions

// ✅ Replace with:
// - ReactMarkdown for display
// - Direct R2 download links
// - Simple API calls
```

## Benefits of New Approach

✅ **Clean Architecture**: No complex JSON parsing  
✅ **Professional PDFs**: Pandoc generates publication-quality documents  
✅ **Reliable Downloads**: R2 presigned URLs, no client-side generation  
✅ **Faster Performance**: Server-side conversion, cached in R2  
✅ **Consistent Formatting**: Same markdown source for all formats  
✅ **Error Resilience**: Fallback strategies built into pipeline  
✅ **Maintainable Code**: Much simpler frontend logic  

## Migration Steps

1. **Test Backend**: Run `python test_markdown_pipeline.py`
2. **Update API Service**: Add new `generateMarkdownDocument()` method
3. **Update UI Components**: Replace complex rendering with markdown display
4. **Remove Old Code**: Delete client-side PDF generation
5. **Test End-to-End**: Verify downloads work from R2
6. **Deploy**: Much more reliable system!

The new approach eliminates all the complex content structure issues and provides professional-quality document generation with reliable downloads.