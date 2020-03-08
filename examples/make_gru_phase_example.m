clear;close all;
fs = 500;
% f_size = [30, 6]; es = '';
f_size = [18.1, 4]; es ='';

T = 3600;
rng(0);
t = 0:1/fs:T - (1/fs);
y_spike = zeros(size(t));
n = pinknoise(length(t));
y_noise_low = 20 * n;
y_noise_hig = 125 * n;
f_alpha = 10;
y_alpha = 10*sin(2*pi*f_alpha*t - rand*2*pi);
t_short = 0:1/fs:2 - (1/fs);
set(0, 'DefaultFigureRenderer', 'painters');

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
% yellow = [236, 225, 51]/255;
yellow = cmap_full(2, :);

cmap = [blue; green; orange; yellow];

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
es = '_erp';
if strcmpi(es, '_erp')
    ix_spike = cumsum(round(3*fs + randi(2*fs, [1, 2000])));
    assert(ix_spike(end)>length(y_spike), '@$@$');
    ix_spike(ix_spike > length(y_spike) - 5*fs) = [];
    y_spike(ix_spike) = 1;
    y_spike = filter(10 * exp(-t_short/0.2).*(1-exp(-t_short/0.015)),1, y_spike);
    %     y_spike = y_spike + 15 * randn(size(y_spike));  % not sure why I had this
elseif strcmpi(es, '_spike')
    ix_spike = cumsum(round(5*fs + randi(fs, [1, 2000])));
    assert(ix_spike(end)>length(y_spike), '@$@$');
    ix_spike(ix_spike > length(y_spike) - 5*fs) = [];
    y_spike(ix_spike) = 1;
    y_spike = filter(500 * exp(-t_short/0.025),1, y_spike);
	%     y_spike = y_spike + 15 * randn(size(y_spike));  % not sure why I had this
end
xlim_ = [9.5 10.5];

f_order = 150;
h = fdesign.bandpass('n,fc1,fc2', f_order, 8, 13, fs);
Hd = design(h, 'window');
y = y_alpha + y_noise_hig + y_spike;
y_sm = smooth(y, 5);%just smooth a little for the plot

y_filter = filter(Hd.Numerator, 1, y);
y_filtfilt = filtfilt(Hd.Numerator, 1, y);

y_h = hilbert(y_filtfilt);
h_f = figure('Name', sprintf(['f__' es]));clf;
clf;hold all;plot(y);plot(real(y_h));plot(imag(y_h));
plot(y_spike, 'k');

%%
ww = fs;
[p_y, f] = pwelch(y, ww, 1, [], fs);
[p_y_filtfilt, f] = pwelch(y_filtfilt, ww, 1, [], fs);
%%
% figure(2);clf;
h_f = figure('Name', sprintf(['fxx' es]));clf;

m = ix_spike' + [0];
w = [-100:0.75 * fs];
t_sec = w/fs;
y_mat = y(m+w);
y_mat_sm = y_sm(m+w);
phi = wrapToPi(angle(y_h(ix_spike)));
circ_rtest(phi)
% abs(mean(exp(1i * phi)))
subplot(1, 4, 2);hold on;
mea_y_mat = mean(y_mat, 1);
std_y_mat = std(y_mat, 0, 1);
sem_y_mat = std_y_mat./sqrt(sum(isfinite(y_mat)));
ciplot(mea_y_mat - sem_y_mat, mea_y_mat + sem_y_mat, t_sec, green, 0.1);
plot(t_sec, mea_y_mat, 'Color', green, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Amplitude (uV)');
axis tight;
make_like_sns(gca, style);

subplot(1, 4, 4);
h = histogram(phi, [-pi:pi/4:pi]);
h.FaceColor = green;
h.EdgeColor = [1, 1, 1];
% title(sprintf('Rayleigh p: %0.3f', circ_rtest(phi)));
xlabel('Phase (rad.)');
xticks([-pi, -pi/2, 0, pi/2, pi]);
xticklabels({'-\pi', '-\pi/2', '0', '\pi/2', '\pi'})
make_like_sns(gca, style);



subplot(1, 4, 3);cla;hold all;
for ix = 2:5
    plot(t_sec, ix*50 + y_mat_sm(ix, :), 'Color', cmap(mod(ix-1, length(cmap))+1, :), 'lineWidth', 1.1)
end
axis tight;
set(gca,'ytick',[]);
set(gca,'ycolor',[1 1 1])

plot([0 0], get(gca, 'ylim'), '--', 'lineWidth', 2, 'Color', 'k');
xlabel('Time (s)');
make_like_sns(gca, style);


subplot(1, 4, 1);cla;hold all;

plot(f, 10*log10(p_y), 'Color', green, 'LineWidth', 2);
% plot(f, 10*log10(p_y_filtfilt), 'r--');
xlim([0 40])
ylim([0 Inf])
xlabel('Frequency (Hz)')
ylabel('PSD (db/Hz)')
xticks([0:10:40]);
make_like_sns(gca, style);

% printForPub(gcf, ['fxx' es], 'fformat', fformat, 'physicalSizeCM', f_size);
print_local(h_f, f_size);

% edit 
% %%
% figure(1);clf;hold on;
% plot(t, 5  + y_alpha + y_noise_low, 'b');
% xlim(xlim_)
% xlabel('Time (s)')
% ylabel('Amplitude (uV)')
% % printForPub(gcf, ['f01' es],'fformat', fformat,'physicalSizeCM',f_size);
%
%
% figure(2);clf;hold on;
% plot(t, y, 'r');
% xlim(xlim_)
% xlabel('Time (s)')
% ylabel('Amplitude (uV)')
% % printForPub(gcf, ['f02' es],'fformat', fformat,'physicalSizeCM',f_size);
%
% figure(3);clf;hold on;
% plot(t, y, 'r');
% plot(t, y_filter, 'b', 'lineWidth', 2);
% xlim(xlim_)
% xlabel('Time (s)')
% ylabel('Amplitude (uV)')
% % printForPub(gcf, ['f03' es],'fformat', fformat,'physicalSizeCM',f_size);
%
%
% figure(4);clf;hold on;
% plot(t, y, 'r');
% plot(t, y_filtfilt, 'k', 'lineWidth', 2);
% xlim(xlim_)
% xlabel('Time (s)')
% ylabel('Amplitude (uV)')
% % printForPub(gcf, ['f04' es],'fformat', fformat,'physicalSizeCM',f_size);
%
% figure(5);clf;hold on;
% plot(t, y_filtfilt, 'k', 'lineWidth', 2);
% plot(t, imag(y_h), 'g', 'lineWidth', 2);
% xlim(xlim_)
% ylim([-20 20])
% xlabel('Time (s)')
% ylabel('Amplitude (uV)')
% % printForPub(gcf, ['f05' es],'fformat', fformat,'physicalSizeCM',f_size);
%
% figure(6);clf;hold on;
% plot(t, angle(y_h), 'k', 'lineWidth', 2);
% xlim(xlim_)
% xlabel('Time (s)')
% ylabel('Angle (rad.)')
% ylim([-pi pi])
% % printForPub(gcf,['f06' es],'fformat', fformat,'physicalSizeCM',f_size);
%
% figure(7);clf;hold on;
% plot(f, 10*log10(p_y), 'r');
% % plot(f, 10*log10(p_y_filtfilt), 'r--');
% xlim([0 50])
% ylim([0 Inf])
% xlabel('Frequency Hz')
% ylabel('PSD (db/Hz)')
% % printForPub(gcf,['f07' es],'fformat', fformat,'physicalSizeCM',[8, f_size(2)]);
%
% figure(8);clf;hold on;
% plot(f, 10*log10(p_y), 'r');
% plot(f, 10*log10(p_y_filtfilt), 'k', 'lineWidth', 2);
% xlim([0 50])
% ylim([0 Inf])
% xlabel('Frequency Hz')
% ylabel('PSD (db/Hz)')
% % printForPub(gcf,['f08' es],'fformat', fformat,'physicalSizeCM',[8, f_size(2)]);
%
% figure(9);clf;hold on;
% plot(t, y_alpha, 'lineWidth', 3);
% xlim([0 0.2]);
% axis off;
% set(gca,'color','none')
% % printForPub(gcf,['f09' es],'fformat', fformat,'physicalSizeCM',[8, f_size(2)]);
%
% figure(10);clf;hold on;
% plot(t, y, 'r');
% % plot(t, real(y_h), 'k', 'lineWidth', 2);
% % plot(t, imag(y_h), 'g', 'lineWidth', 2);
% xlim(xlim_)
% ylim([- 100 100])
% xlabel('Time (s)')
% ylabel('Amplitude (uV)')
% % printForPub(gcf, ['f10' es],'fformat', fformat,'physicalSizeCM',f_size);
%
% close all;