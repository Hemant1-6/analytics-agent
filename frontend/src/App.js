import React, { useState, useRef } from 'react';
import { BarChart, Bar, PieChart, Pie, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { Upload, MessageSquare, Send, Loader, FileText, BarChart2, PieChart as PieIcon, AlertTriangle } from 'lucide-react';

// --- Helper Components ---
const Icon = ({ name }) => {
  switch (name) {
    case 'VBC': case 'VDBC': case 'VSBC': return <BarChart2 className="h-6 w-6 text-gray-500" />;
    case 'PIE': return <PieIcon className="h-6 w-6 text-gray-500" />;
    case 'TBL': case 'LDRBRD': return <FileText className="h-6 w-6 text-gray-500" />;
    default: return <BarChart2 className="h-6 w-6 text-gray-500" />;
  }
};

const LoadingSpinner = () => (
  <div className="flex justify-center items-center p-4">
    <Loader className="h-8 w-8 text-indigo-500 animate-spin" />
    <p className="ml-2 text-gray-600">Analyzing your data...</p>
  </div>
);

const ErrorDisplay = ({ error }) => (
    <div className="bg-red-50 border-l-4 border-red-400 p-4 mt-4 rounded-r-lg">
        <div className="flex">
            <div className="py-1"><AlertTriangle className="h-6 w-6 text-red-500 mr-3" /></div>
            <div>
                <p className="font-bold text-red-800">An Error Occurred</p>
                <p className="text-sm text-red-700">{error}</p>
            </div>
        </div>
    </div>
);

// --- Chart Rendering Components ---
const COLORS = ["#05abf3", "#f3b4b7", "#fdbf16", "#4db3e5", "#756fd7", "#e189b5"];

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white p-2 border border-gray-300 rounded shadow-lg">
        <p className="label font-bold text-gray-700">{`${label}`}</p>
        {payload.map((entry, index) => (
          <p key={`item-${index}`} style={{ color: entry.color }}>
            {`${entry.name}: ${typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value}`}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

const renderChart = (config) => {
    if (!config || !config.widget_data) return <p>No chart data available.</p>;

    const { widget_typeofchart } = config.widget;
    const { labels, datasets, columns, records } = config.widget_data;

    switch (widget_typeofchart) {
        case 'VBC': case 'VDBC': case 'VSBC':
            const chartData = labels.map((label, index) => {
                const dataEntry = { name: label };
                datasets.forEach(dataset => { dataEntry[dataset.label] = dataset.data[index]; });
                return dataEntry;
            });
            return (
                <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend />
                        {datasets.map((dataset, index) => (
                            <Bar key={dataset.label} dataKey={dataset.label} fill={dataset.backgroundColor || COLORS[index % COLORS.length]} />
                        ))}
                    </BarChart>
                </ResponsiveContainer>
            );
        
        case 'PIE':
            const pieData = labels.map((label, index) => ({ name: label, value: datasets[0].data[index] }));
            return (
                <ResponsiveContainer width="100%" height={400}>
                    <PieChart>
                        <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={150} fill="#8884d8" label>
                            {pieData.map((entry, index) => (<Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />))}
                        </Pie>
                        <Tooltip content={<CustomTooltip />} />
                        <Legend />
                    </PieChart>
                </ResponsiveContainer>
            );

        case 'TBL': case 'LDRBRD':
             return (
                <div className="overflow-x-auto">
                    <table className="min-w-full bg-white border border-gray-200">
                        <thead className="bg-gray-50">
                            <tr>{columns.map(col => <th key={col.key} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{col.label}</th>)}</tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            {records.map((record, rowIndex) => (
                                <tr key={rowIndex} className="hover:bg-gray-50">
                                    {columns.map(col => <td key={`${rowIndex}-${col.key}`} className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{record[col.key]}</td>)}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            );
        
        case 'CARD':
            return (
                 <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 text-center">
                    <h3 className="text-lg font-medium text-gray-500">{records[0].label}</h3>
                    <p className="mt-2 text-4xl font-bold text-gray-900">
                        {typeof records[0].data === 'number' ? records[0].data.toLocaleString() : records[0].data}
                    </p>
                </div>
            )
        default: return <p>Unsupported chart type: {widget_typeofchart}</p>;
    }
};

// --- Main App Component ---
export default function App() {
  const [file, setFile] = useState(null);
  const [datasetId, setDatasetId] = useState(null);
  const [query, setQuery] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      handleFileUpload(selectedFile);
    }
  };

  const handleFileUpload = async (selectedFile) => {
    if (!selectedFile) return;
    setIsLoading(true);
    setError(null);
    setDatasetId(null);
    setChatHistory([]);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/api/analytics/upload', { method: 'POST', body: formData });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'File upload failed');
      }
      const data = await response.json();
      setDatasetId(data.dataset_id);
      setChatHistory([{ type: 'system', message: `Successfully uploaded '${data.dataset_id}'. You can now ask questions.` }]);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || !datasetId || isLoading) return;

    setIsLoading(true);
    setError(null);
    
    const userMessage = { type: 'user', message: query };
    setChatHistory(prev => [...prev, userMessage]);
    const currentQuery = query;
    setQuery('');

    try {
      const response = await fetch('/api/analytics/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: datasetId, query: currentQuery }),
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Analysis failed');
      }
      const data = await response.json();
      const botMessage = { type: 'bot', data };
      setChatHistory(prev => [...prev, botMessage]);
    } catch (err) {
      setError(err.message);
      const errorMessage = { type: 'system', message: `Error: ${err.message}` };
      setChatHistory(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleSuggestionClick = (suggestion) => {
    setQuery(suggestion);
    const input = document.getElementById('query-input');
    if(input) input.focus();
  }

  const onDragOver = (e) => e.preventDefault();
  const onDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
        setFile(droppedFile);
        handleFileUpload(droppedFile);
    }
  };

  return (
    <div className="flex h-screen bg-gray-100 font-sans">
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-100 p-4 md:p-8">
            <div className="max-w-7xl mx-auto">
                <h1 className="text-3xl font-bold text-gray-800 mb-6">Conversational Analytics Agent</h1>
                <div className="bg-white p-6 rounded-xl shadow-md mb-8">
                    <div 
                        className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-indigo-500 transition-colors"
                        onClick={() => fileInputRef.current.click()}
                        onDragOver={onDragOver}
                        onDrop={onDrop}
                    >
                        <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept=".csv" />
                        <div className="flex flex-col items-center text-gray-500">
                            <Upload className="h-12 w-12 mb-4" />
                            {file ? (
                                <><p className="font-semibold text-gray-700">{file.name}</p><p className="text-sm">{datasetId ? "Ready to analyze." : "Uploading..."}</p></>
                            ) : (
                                <><p className="font-semibold text-gray-700">Drag & drop your CSV file here</p><p className="text-sm">or click to select a file</p></>
                            )}
                        </div>
                    </div>
                </div>
                <div className="space-y-6">
                    {chatHistory.map((item, index) => (
                        <div key={index}>
                            {item.type === 'user' && (<div className="flex justify-end"><div className="bg-indigo-500 text-white p-4 rounded-xl max-w-2xl">{item.message}</div></div>)}
                            {item.type === 'bot' && item.data && (
                                <div className="bg-white p-6 rounded-xl shadow-md">
                                    <div className="flex items-center mb-4">
                                        <Icon name={item.data.chart_type} />
                                        <h2 className="ml-3 text-xl font-semibold text-gray-800">{item.data.suggested_chart_config.widget.widget_title}</h2>
                                    </div>
                                    <p className="text-gray-600 mb-6">{item.data.description}</p>
                                    <div className="mb-6">{renderChart(item.data.suggested_chart_config)}</div>
                                    {item.data.proactive_suggestions && item.data.proactive_suggestions.length > 0 && (
                                        <div>
                                            <h4 className="font-semibold text-gray-700 mb-2">Suggested Next Steps:</h4>
                                            <div className="flex flex-wrap gap-2">
                                                {item.data.proactive_suggestions.map((s, i) => (
                                                    <button key={i} onClick={() => handleSuggestionClick(s)} className="bg-indigo-100 text-indigo-700 px-3 py-1 rounded-full text-sm hover:bg-indigo-200 transition-colors">{s}</button>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                            {item.type === 'system' && (<div className="text-center text-sm text-gray-500 py-2">{item.message}</div>)}
                        </div>
                    ))}
                    {isLoading && <LoadingSpinner />}
                    {error && <ErrorDisplay error={error} />}
                </div>
            </div>
        </main>
        <footer className="bg-white border-t border-gray-200 p-4">
            <div className="max-w-7xl mx-auto">
                <form id="query-form" onSubmit={handleQuerySubmit} className="flex items-center">
                    <MessageSquare className="h-6 w-6 text-gray-400 mr-3" />
                    <input
                        id="query-input"
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder={datasetId ? "Ask a question about your data..." : "Please upload a file first"}
                        disabled={!datasetId || isLoading}
                        className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-shadow"
                    />
                    <button type="submit" disabled={!datasetId || isLoading} className="ml-4 bg-indigo-600 text-white p-3 rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 transition-colors">
                        <Send className="h-6 w-6" />
                    </button>
                </form>
            </div>
        </footer>
      </div>
    </div>
  );
}
