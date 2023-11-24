#ifdef USE_PIX
#define IS_PIX_ENABLED 1
extern void CPIXInitModule(void);
extern void CPIXSetEnableHUD(BOOL is_enabled);
extern void CPIXBeginEventCL(ID3D12GraphicsCommandList*, BYTE category, char const* name);
extern void CPIXEndEventCL(ID3D12GraphicsCommandList*);
extern void CPIXBeginEventCQ(ID3D12CommandQueue*, BYTE category, char const* name);
extern void CPIXEndEventCQ(ID3D12CommandQueue*);
extern void CPIXBeginEvent(BYTE category, char const* name);
extern void CPIXEndEvent(void);
extern void CPIXSetMarkerCL(ID3D12GraphicsCommandList*, BYTE category, char const* name);
extern void CPIXSetMarkerCQ(ID3D12CommandQueue*, BYTE category, char const* name);
extern void CPIXSetMarker(BYTE category, char const* name);
extern void CPIXReportCounter(wchar_t const* name, float value);
#else
#define IS_PIX_ENABLED 0
inline void CPIXInitModule(void) {}
inline void CPIXSetEnableHUD(BOOL is_enabled) {}
inline void CPIXBeginEventCL(ID3D12GraphicsCommandList*, BYTE category, char const* name) {}
inline void CPIXEndEventCL(ID3D12GraphicsCommandList*) {}
inline void CPIXBeginEventCQ(ID3D12CommandQueue*, BYTE category, char const* name) {}
inline void CPIXEndEventCQ(ID3D12CommandQueue*) {}
inline void CPIXBeginEvent(BYTE category, char const* name) {}
inline void CPIXEndEvent(void) {}
inline void CPIXSetMarkerCL(ID3D12GraphicsCommandList*, BYTE category, char const* name) {}
inline void CPIXSetMarkerCQ(ID3D12CommandQueue*, BYTE category, char const* name) {}
inline void CPIXSetMarker(BYTE category, char const* name) {}
inline void CPIXReportCounter(wchar_t const* name, float value) {}
#endif
