function st = viz_style()
% Returns colours as RGB rows in [0,1], font names, type sizes (pt) and animation settings.
hex = @(h) sscanf(h(2:end), '%2x')' / 255;
st.ink = hex('#0b0b0b');
st.ink2 = hex('#52514e');
st.muted = hex('#898781');
st.grid = hex('#e1e0d9');
st.axis = hex('#c3c2b7');
st.surface = hex('#ffffff');
st.blue = hex('#2a78d6');
st.orange = hex('#eb6834');
st.aqua = hex('#1baf7a');
st.ell3 = hex('#cde2fb');
st.ell2 = hex('#9ec5f4');
st.ell1 = hex('#6da7ec');
st.font = 'Segoe UI';
st.font_title = 'Segoe UI Semibold';
st.fs_title = 25;
st.fs_sub = 17;
st.fs_panel = 18;
st.fs_label = 16;
st.fs_tick = 15;
st.fs_note = 14;
st.lw = 2.8;
st.fig_px = [1280 720];
st.fps = 30;
end
