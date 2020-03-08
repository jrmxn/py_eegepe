clear;
fs = 500;
f_size = [15, 2.5]; es ='';
T = 20;
rng(0);
t = 0:1/fs:T - (1/fs);
y_spike = zeros(size(t));
n = pinknoise(length(t));
y_noise_low = 20 * n;
y_noise_hig = 125 * n;
f_alpha = 10;
y_alpha = 10*sin(2*pi*f_alpha*t);

[~, ix_spike] = min(abs(t-10.05));
es = '_spike';
y_spike(ix_spike) = 1;
y_spike = filter(300*exp(-t/0.01),1, y_spike);

xlim_ = [9.5 10.2];

f_order = 150;
h = fdesign.bandpass('n,fc1,fc2', f_order, 8, 12, fs);
Hd = design(h, 'window');
y = y_alpha + y_noise_hig + y_spike;

y_filter = filter(Hd.Numerator, 1, y);
y_filtfilt = filtfilt(Hd.Numerator, 1, y);

y_h = hilbert(y_filtfilt);

w = 2*fs;
[p_y, f] = pwelch(y, w, 1, [], fs);
[p_y_filtfilt, f] = pwelch(y_filtfilt, w, 1, [], fs);
%%
cmap_full = [ ...
    [0.00392156862745098, 0.45098039215686275, 0.6980392156862745]; ...
    [0.8705882352941177, 0.5607843137254902, 0.0196078431372549]; ...
    [0.00784313725490196, 0.6196078431372549, 0.45098039215686275]; ...
    [0.8352941176470589, 0.3686274509803922, 0.0]; ...
    [0.8, 0.47058823529411764, 0.7372549019607844]; ...
    ];

backg = [234, 234, 242]/255;
blue = cmap_full(1, :);
green = cmap_full(3, :);
orange = cmap_full(4, :);

% style = 'default';
style = 'ticks';

v.figformat = 'svg';
v.figdir = pwd;
v.figsave = true;
v.fontsizeAxes = 7;
v.fontsizeText = 8;


% if not(exist(p_fig, 'dir')==7), mkdir(p_fig);end
%
% if not(vobs.figsave)
%     warning('Not saving figures!');
% end
print_local = @(h, dim) printForPub(h, sprintf('f_%s', h.Name), 'doPrint', v.figsave,...
    'fformat', v.figformat , 'physicalSizeCM', dim, 'saveDir', v.figdir, ...
    'fontsizeText', v.fontsizeText, 'fontsizeAxes', v.fontsizeAxes);

%%
% figure(1);clf;hold on;
% plot(t, 5  + y_alpha + y_noise_low, 'b');
% xlim(xlim_)
% xlabel('Time (s)')
% ylabel('Amplitude (uV)')
% printForPub(gcf, ['f01' es],'fformat','svg','physicalSizeCM',f_size);


h_f = figure('Name', sprintf('02'));clf;
plot(t, y, 'Color', green, 'lineWidth', 2);
xlim(xlim_)
xlabel('Time (s)')
ylabel('Amp. (uV)')

make_like_sns(gca, style);
print_local(h_f, f_size);

% figure(3);clf;hold on;
% plot(t, y, 'r');
% plot(t, y_filter, 'b', 'lineWidth', 2);
% xlim(xlim_)
% xlabel('Time (s)')
% ylabel('Amplitude (uV)')
% printForPub(gcf, ['f03' es],'fformat','svg','physicalSizeCM',f_size);


% figure(4);clf;hold on;
% plot(t, y, 'r');
% plot(t, y_filtfilt, 'k', 'lineWidth', 2);
% xlim(xlim_)
% xlabel('Time (s)')
% ylabel('Amplitude (uV)')
% printForPub(gcf, ['f04' es],'fformat','svg','physicalSizeCM',f_size);

h_f = figure('Name', sprintf('05'));clf;hold on;
ix = ix_spike - (length(Hd.Numerator)-1)/2;
plot(t(1:ix_spike), y_filtfilt(1:ix_spike), 'k--' ,'lineWidth', 2);
plot(t(1:ix_spike), imag(y_h(1:ix_spike)), '--', 'Color', blue, 'lineWidth', 2);
plot(t(1:ix), y_filtfilt(1:ix), 'k', 'lineWidth', 2);
plot(t(1:ix), imag(y_h(1:ix)), '-', 'Color', blue, 'lineWidth', 2);
xlim(xlim_)
ylim([-20 20])
xlabel('Time (s)')
ylabel('Amp. (uV)')

make_like_sns(gca, style);
print_local(h_f, f_size);

h_f = figure('Name', sprintf('alpha_example'));clf;
h = fdesign.bandpass('n,fc1,fc2', 200, 8, 12, 1000);
Hd = design(h, 'window');
% orange = [213, 94, 0]/255;
plot(Hd.Numerator, 'color', orange, 'lineWidth', 2)
axis tight;
ax = gca;
set(ax,'box','off')
set(ax,'xcolor','w','ycolor','w')
% printForPub(gcf, ['alpha_example'], 'fformat', 'svg', 'physicalSizeCM', [5, 1]);
print_local(h_f, [5, 1]);


% figure(6);clf;hold on;
% plot(t, angle(y_h), 'k', 'lineWidth', 2);
% xlim(xlim_)
% xlabel('Time (s)')
% ylabel('Angle (rad.)')
% ylim([-pi pi])
% printForPub(gcf,['f06' es],'fformat','svg','physicalSizeCM',f_size);
%
% figure(7);clf;hold on;
% plot(f, 10*log10(p_y), 'r');
% % plot(f, 10*log10(p_y_filtfilt), 'r--');
% xlim([0 50])
% ylim([0 Inf])
% xlabel('Frequency Hz')
% ylabel('PSD (db/Hz)')
% printForPub(gcf,['f07' es],'fformat','svg','physicalSizeCM',[8, f_size(2)]);
%
% figure(8);clf;hold on;
% plot(f, 10*log10(p_y), 'r');
% plot(f, 10*log10(p_y_filtfilt), 'k', 'lineWidth', 2);
% xlim([0 50])
% ylim([0 Inf])
% xlabel('Frequency Hz')
% ylabel('PSD (db/Hz)')
% printForPub(gcf,['f08' es],'fformat','svg','physicalSizeCM',[8, f_size(2)]);
%
% figure(9);clf;hold on;
% plot(t, y_alpha, 'lineWidth', 3);
% xlim([0 0.2]);
% axis off;
% set(gca,'color','none')
% printForPub(gcf,['f09' es],'fformat','svg','physicalSizeCM',[8, f_size(2)]);
%
% figure(10);clf;hold on;
% plot(t, y, 'r');
% % plot(t, real(y_h), 'k', 'lineWidth', 2);
% % plot(t, imag(y_h), 'g', 'lineWidth', 2);
% xlim(xlim_)
% ylim([- 100 100])
% xlabel('Time (s)')
% ylabel('Amplitude (uV)')
% printForPub(gcf, ['f10' es],'fformat','svg','physicalSizeCM',f_size);

% close all;