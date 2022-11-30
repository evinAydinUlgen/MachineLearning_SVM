# Makine Öğrenmesi Sınıflandırma Algoritması : Destek Vektör Makineleri Raporu
## İÇİNDEKİLER
* [SINIFLANDIRMA](https://github.com/evinAydinUlgen/MachineLearning_SVM#siniflandirma)
  * [Makine Öğrenmesi](https://github.com/evinAydinUlgen/MachineLearning_SVM#makine-%C3%B6%C4%9Frenmesi)
  * [Classification(Sınıflandırma) Nedir?](https://github.com/evinAydinUlgen/MachineLearning_SVM/blob/main/README.md#classification-s%C4%B1n%C4%B1fland%C4%B1rma-nedir)
    * [Sınıflandırma Türleri](https://github.com/evinAydinUlgen/MachineLearning_SVM/blob/main/README.md#s%C4%B1n%C4%B1fland%C4%B1rma-t%C3%BCrleri)
      * [İkili Sınıflandırma](https://github.com/evinAydinUlgen/MachineLearning_SVM/blob/main/README.md#i%CC%87kili-s%C4%B1n%C4%B1fland%C4%B1rma)
      * [Çok Sınıflı Sınıflandırma](https://github.com/evinAydinUlgen/MachineLearning_SVM/blob/main/README.md#%C3%A7ok-s%C4%B1n%C4%B1fl%C4%B1-s%C4%B1n%C4%B1fland%C4%B1rma)
  * [Sınıflandırma Algoritmaları](https://github.com/evinAydinUlgen/MachineLearning_SVM/blob/main/README.md#s%C4%B1n%C4%B1fland%C4%B1rma-algoritmalar%C4%B1)
  * [Sınıflandırma Örneği: Meyveler Örneği](https://github.com/evinAydinUlgen/MachineLearning_SVM/blob/main/README.md#s%C4%B1n%C4%B1fland%C4%B1rma-%C3%B6rne%C4%9Fi)
   * [Model Başarısı Değerlendirme-Sınıflandırma](https://github.com/evinAydinUlgen/MachineLearning_SVM/blob/main/README.md#model-ba%C5%9Far%C4%B1s%C4%B1-de%C4%9Ferlendirme-s%C4%B1n%C4%B1fland%C4%B1rma)
* [DESTEK VEKTÖR MAKİNELERİ](https://github.com/evinAydinUlgen/MachineLearning_SVM/edit/main/README.md#destek-vekt%C3%B6r-maki%CC%87neleri%CC%87)
  * [Tarihçe](https://github.com/evinAydinUlgen/MachineLearning_SVM/edit/main/README.md#tarih%C3%A7e)
  * [Destek Vektör Makineleri Nedir?](https://github.com/evinAydinUlgen/MachineLearning_SVM/edit/main/README.md#destek-vekt%C3%B6r-makineleri-nedir)
  * [Destek Vektör Makineleri Nasıl Çalışır?](https://github.com/evinAydinUlgen/MachineLearning_SVM/edit/main/README.md#destek-vekt%C3%B6r-makineleri-nas%C4%B1l-%C3%A7al%C4%B1%C5%9F%C4%B1r)
  * [DVM Avantaj Ve Dezavantajları](https://github.com/evinAydinUlgen/MachineLearning_SVM/edit/main/README.md#dvm-avantaj-ve-dezavantajlar%C4%B1)
* [DOĞRUSAL OLAN\OLMAYAN DVM](https://github.com/evinAydinUlgen/MachineLearning_SVM/edit/main/README.md#do%C4%9Frusal-olanolmayan-dvm)
  * [Doğrusal Olan Destek Vektör Makineleri](https://github.com/evinAydinUlgen/MachineLearning_SVM/edit/main/README.md#1do%C4%9Frusal-destek-vekt%C3%B6r-makineleri)
  * [Doğrusal Olmayan Destek Vektör Makineleri](https://github.com/evinAydinUlgen/MachineLearning_SVM/edit/main/README.md#2do%C4%9Frusal-olmayan-destek-vekt%C3%B6r-makineleri)
* [DVM KULLANIM ALANLARI](https://github.com/evinAydinUlgen/MachineLearning_SVM/edit/main/README.md#dvm-kullanim-alanlari)
* [KAYNAKÇA](https://github.com/evinAydinUlgen/MachineLearning_SVM/edit/main/README.md#kaynak%C3%A7a)

## SINIFLANDIRMA
### Makine Öğrenmesi 
*Makine öğrenmesi* yapay zekanın bir alt dalı olan bilim dalıdır.Amacı, matematiksel veya istatistiksel işlemler ile veriler üzerinden çıkarımlar yaparak tahminlerde bulunan sistemler oluşturmakır.Günümüzde birçok makine Öğrenmesi metodu bulunmaktadır.Bir makine öğrenmesi metodu tahminde bulunmak için bir çıktı üretmek zorundadır:
Eğer bu çıktı kategorik ise ***sınıflandırma***,nümerik ise ***regresyon*** adını almaktadır.Açıklayıcı bir modelleme olan kümeleme ***clustering*** ise benzer gözlemleri aynı kümelere atama işlemidir. ***Birliktelik Kuralları (Association Rules)*** ile gözlemler arasındaki ilginç bağlantılar bulunabilir. Sınıflandırma ve regresyon yöntemleri eğitim verisine ihtiyaç duyduğu için supervised öğrenme olarak anlatılırken diğer iki metot eğitim verisi gerektirmediği için unsupervised öğrenme olarak adlandırılır.<br><br>
![AI](https://miro.medium.com/max/443/1*TBT5QlNAkEbggyN8jDp6sw.png)

### Classification (Sınıflandırma) Nedir?
Makinenin öğrenmesi için girdi ve çıktıların birlikte yer aldığı örneklerin makineye sunulduğu öğrenme biçimi gözetimli öğrenme olarak ifade edilmişti. Her girdi vektörünü, sonlu sayıdaki bir ayrık kategoriye atamayı amaçlayan durumlar ise sınıflandırma (classification) problemi olarak ele alınmaktadır. Sınıflandırma  problemlerinde çıktı uzayındaki her bir eleman birer sınıf (class), sınıflandırma problemini algoritmaya da sınıflandırıcı (classifier) adı verilmektedir.<br>
Bir sınıflandırıcı, **ĉ: X -> C, C ={C1, C2, …, Cn}**  biçiminde gösterilebilir .Burada sınıflandırıcının **ĉ(x)** şeklinde ifade edilmesinin nedeni, gerçekte var olan ancak sınıfı bilinmeyen  **c(x)** fonksiyonunun tahminini ifade etmesidir. Bir sınıflandırıcı için örnekler **(x, c(x))** formunu almakta, **c(x)** ise gerçek sınıf değerini göstermektedir.  Sınıflandırma,  makine  öğrenmesinin  popüler  ve  temel  görevlerinden  biridir  ve bilinmeyen  bir  veri  parçasının  bilinen  bir  gruba  yerleştirilmesinde kullanılmaktadır.  Örneklerden çıkarım olarak da bilinen sınıflandırmada amaç, kavram tanımı elde edildikten sonra, daha önce algoritmaya  tanıtılmamış örnekleri en yüksek doğrulukla etiketleyecek sınıflandırıcıyı geliştirmektir. Makine öğrenmesi sınıflandırması literatürde örüntü tanıma olarak da  adlandırılmaktadır .<br>
Sınıflandırmada bazı modeller (olasılı  (probabilistic)  sınıflandırıcı)  çıktı olarak bir örneğin  belirli  bir  sınıfa  aitliğini  gösteren  olasılık  değerini  verirken,  bazıları  (ayrık (discrete)  sınıflandırıcı)  sadece  örneğin  tahmin  edilen  sınıfını  gösteren  ayrık  sınıf etiketini vermektedir.  

#### Sınıflandırma Türleri
Makine öğrenmesi verileri gruplara, sınıflara ve parçalara göre sınıflar. Sınıflandırma problemlerinde etiket verileri kategoriktir ve önceden kategorize edilmiştir.<br> Sınıflandırma algoritmaları, kendisine verilen gözlemin, hangi sınıfa ait olduğunu bulmaya yarar.<br>
Sınıflandırmada kendi içerisinde ikiye ayrılır:<br>
1.İkili Sınıflandırma - Binary Classification<br>
2.Çok Sınıflı Sınıflandırma - Multi-class Classification<br>
>Buradaki sınıflar, verisetimizde ki bağımlı değişkene ithafen kullanılmaktadır. Bir bağımlı değişken içerisinde kaç farklı eşsiz değer varsa o kadar sınıf vardır demek çokta yanlış olmaz.<br>
##### İkili Sınıflandırma
İkili sınıflandırmada, sınıflarımızın sayısı ikidir, sınıflarımızı ikili gruplara ayırırız. Etiketlerimiz ikili değerler alır, kadın veya erkek, pozitif veya negatif gibi verileri 0 veya 1 gibi değerlerle gösteririz. Bu sınıflar "pozitif sınıf" ve "negatif sınıf" olarak ikiye ayrılır. Sınıflandırma problemleri arasında en basit olan sınıflandırma problemidir.<br>
Bazı kullanım alanları:<br>
+Bir hastanın spesifik bir hastalığa sahip olup olmadığının tespit edilmesi.<br>
+Bir ihtiyacın karşılanıp karşılanmadığına yönelik kalite kontrol uygulamaları.<br>
+Mailin spam olup olmadığını tespit edilmesi.<br>
+"Bir kişi kredi borcunu ödeyebilir mi, ödeyemez mi?" sorusunun cevabının tespit edilmesi.<br>
+"Online alışveriş sitesini ziyaret eden kullanıcı, ziyaret ettiği ürünü alır mı almaz mı?" sorusunun cevabının tespit edilmesi.<br>
+Ses verisinde ki sesin sahibinin cinsiyetinin tespiti.<br>
##### Çok Sınıflı Sınıflandırma
Çoklu sınıflandırmada, sınıflarımızın sayısı en az üçtür, üç veya daha fazla sınıflar söz konusudur.<br>
Bazı kullanım alanları:<br>
+Haberlerin kategorilerinin belirlenmesi.<br>
+Kitapların konularına göre sınıflandırılması.<br>
+Yüz tanıma.<br>
+Bitkilerin türlerinin belirlenmesi. (Örn. Iris Veriseti)<br>
+Optik Karakter Tanıma - Resimlerde ki karakterlerin tespiti. (Örn. MNIST veriseti)<br><br>
![image](https://user-images.githubusercontent.com/106628455/204885501-44ebcb8b-309b-4e77-94bf-5a535c86ae5b.png)<br>

### Sınıflandırma Algoritmaları
 Sınıflandırma algoritmaları, verilen eğitim kümesinden bu dağılım şeklini öğrenirler ve daha sonra sınıfının belirli olmadığı test verileri geldiğinde doğru şekilde sınıflandırmaya çalışırlar.<br>
Veri kümesi üzerinde verilen bu sınıfları belirten değerlere etiket (label) ismi verilir ve gerek eğitim gerekse test sırasında verinin sınıfının belirlenmesi için kullanılırlar.<br>
Literatürde  makine  öğrenmesinde sınıflandırma  için  kullanılan Karar Ağaçları,  Naive Bayes  Sınıflandırıcı,  Yapay Sinir Ağları,  Destek  Vektör  Makineleri,  k-En  Yakın  Komşu  Algoritması  gibi  çok  sayıda algoritma  mevcuttur.<br>  
![image](https://user-images.githubusercontent.com/106628455/204878914-12bd29d8-8f64-4bde-a086-ce376d1df584.png)<br>
>*"Neden bu kadar çok algoritma söz konusu?"* sorusunu sorabilirsiniz, bu sorunun cevabı verisetimizde gizli. Bir verisetinde kullanacağımız tüm sınıflandırma algoritmalarının performansı aynı olmayabilir ki genelde de böyledir, bazı verisetlerinde KNN iyi bir performans gösterirken Lojistik Regresyon kötü bir performans gösterebilir. "Hangi algoritmayı seçmeliyim?", sorusunun cevabı ise verisetiyle ilgilidir, verinizi iyi bir şekilde analiz edip tanımalısınız ancak buda yetmeyebilir, sonuç olarak deneme-yanılma ve pratik ile zamanla algoritma seçme becerinizi geliştirebilirsiniz.

### Sınıflandırma Örneği
Elimizde birçok elma ve portakal resmi bulunsun. Her resimden meyvelerin rengini ve boyutunu belirleyebiliriz. Gözetimli öğrenme için ilk adım etiketli eğitim verilerini edinmek gerekir. Bizim örneğimizde, her biri elma veya portakal olarak etiketlenen çok sayıda meyve resmi elde etmemiz anlamına geliyor. Bu resimlerden, renk ve boy bilgilerini çıkarabilir ve bunları elma ya da portakal olmakla ilişkilendirebiliriz. <br>
Örneğin, etiketli eğitim verimizi aşağıdaki gibi grafiğe dökebiliriz:<br><br>
![image](https://user-images.githubusercontent.com/106628455/204895567-21568c24-b2ee-4dff-b423-17d008e90e2e.png)
<br>
Kırmızı x’ler elma ve turuncu x’ler portakal olarak etiketlenmiş. Görebileceğiniz gibi veri de bir “pattern” var gibi gözüküyor. Kırmızı  x’ler solda ve turuncu x’ler sağda kümelenmiş. Algoritmamızında bu tür “pattern” leri öğrenmesini istiyoruz.<br>
Bu problem için, amacımız, iki etiketli grubu bir karar sınırı (decision boundary) ile ayıracak algoritmayı yazmak. Basit bir karar sınırı aşağıdaki gibi olabilir :<br>
![image](https://user-images.githubusercontent.com/106628455/204896086-a0b6454d-d2dc-466e-887c-dd799fbc04fb.png)
<br><br>
Burada karar sınırı, basit bir çizgi gibi gözüküyor. Daha karmaşık makine öğrenimi algoritmaları aşağıdaki gibi bir çizgi de oluşturabilir:
<br><br>
![image](https://user-images.githubusercontent.com/106628455/204896318-9dea6494-ce36-43b2-9d16-d00dbc257d64.png)
<br><br>
Algoritmamıza eğitim setindeki elma ve portakal etiketli resimleri vererek, algoritma “deneyim”ini, eğitmediğimiz resimler üzerine “genelleştirmek”tedir. Daha açıkça, elimizdeki verileri kullanarak, hiç görmediği resimler üzerinde varsayım yapmaktadır. Örnek olarak, bir meyve resmi verdiğimizde, çizdiğimiz karar sınırına göre resmi portakal olarak sınıflandırabilir:<br><br>
![image](https://user-images.githubusercontent.com/106628455/204896388-eab01a67-59ef-4760-b7b8-40498339ce86.png)
<br>
**Tahmin Sonucu**<br>
Bir eğitim seti üzerinde algoritmamızı çalıştırdık ve yeni veriler üzerine tahminimizi yaptık.<br>
Bu stratejiyi bir çok problem üzerine uygulayabiliriz: Tümörleri iyi/kötü huylu olarak sınıflandırma, bir e-postayı spam/ham olarak sınıflandırma vb… Bu tip makine öğrenmesi – verileri karar sınırları ile ayırma – sınıflandırma olarak adlandırılmaktadır.

### Model Başarısı Değerlendirme-Sınıflandırma
Bir makine öğrenmesi çalışmasında modelin kaç tane durumu doğru olarak tahmin ettiği çok yeterli bir değerlendirme kriteri değildir. Örneğin elimizde 1000 tane veri olsun. 900 tanesi A sınıfı içinde iken 100 tanesi B sınıfının içinde olsun. Bir makine öğrenmesi metodu geliştirirsen ve hepsini A olarak bilirse başarısı %90’dır. Modelimizi daha iyi değerlendirmek için “confusion matrix”e bakalım:<br>
**Confusion Matrix**<br>
Confusion matrix, verideki var olan durum ile sınıflama modelimizin doğru ve yanlış tahminlerinin sayısını gösterir. Aşağıda 2×2’lik bir confusion matrix görülmektedir. Bu matrix tahmindeki hedef sayısına göre değişir. Başka bir değişle NxN’lik olabilir. <br><br>
![image](https://user-images.githubusercontent.com/106628455/204890419-a18ca906-88b4-4884-b321-a1ce30c2b1f2.png)
<br><br>
Kafamız karışabilir:<br>
True-positive - TP - Doğru Tahmin <br>
True-negative - TN - Doğru Tahmin<br>
False-positive - FP - Tip 1 hatası - Yanlış Tahmin<br>
False-negative - FN - Tip 2 hatası - Yanlış Tahmin<br>
Karışmasın..:<br>
TP - Gerçek değer pozitifken, tahmin edilen değer pozitiftir.<br>
FN - Gerçek değer pozitifken, tahmin edilen değer negatiftir.<br>
TN - Gerçek değer negatifken, tahmin edilen değer negatiftir.<br>
FP - Gerçek değer negatifken, tahmin edilen değer pozitiftir.<br>
Burada True ve False değerleri gerçek değerlerini, Positive ve Negative değerleri tahmin değerlerini işaret eder.<br>
<br>
**Recall - Duyarlılık** : Tüm pozitif sınıflardan ne kadar doğru tahmin ettiğimizi ölçer.<br>
`Recall = TP / (TP+FN) `<br>
**Precision - Kesinlik** : Pozitif tahminlerin kaçının gerçek değeri pozitif hesabı yapar.<br>
`Precision = TP / (TP+FP) `<br>
**Accuracy - Doğruluk** : Doğru tahmin sayısının toplam tahmin sayısına bölümünden elde edilir.<br>
`Accuracy = (TP+TN) / (TP+TN+FN+FP)`<br>
**F-Meaure** : Recall ve Precision’ın harmonik ortalaması.<br>
`F-Measure = 2*Precision*Recall / (Precision + Recall)`<br><br><br>


## DESTEK VEKTÖR MAKİNELERİ
### Tarihçe
-Doğrusal Diskriminant Yöntemi- Fisher, 1936<br>
-Perceptron- F.Rosenblatt, 1957<br>
-Genelleştirilmiş Düşey Algoritmaları- Vapnik&Lerner, 1963<br>
-İstatistiksel Öğrenme- Vapnik&Chervonenkis, 1974<br>
-Estimation of Dependences Based of Emprical Data- Vapnik, 1982 (VC Teorisi)<br>
-Kernel Trick- Busor,Guyon&Vapnik, 1992<br>
-Soft Margin Yaklaşımı- Cortes, 1995<br>
<br>
1980 Öncesi:<br>
<br>
– Hemen hemen tüm öğrenme yöntemleri doğrusal karar yüzeylerini öğrendi.<br>
– Doğrusal öğrenme yöntemleri teorik özelliklere sahipti.<br>
<br>
1980<br>
<br>
– Karar ağaçları ve NN’ler, doğrusal olmayan karar yüzeylerinin verimli öğrenilmesine izin verdi.<br>
– Az teorik temel ve hepsi yerel minimadan muzdarip çalışmalar.<br>
<br>
1990’lar<br>
<br>
– Geliştirilmiş hesaplamalı öğrenme teorisine dayanan doğrusal olmayan fonksiyonlar için etkili öğrenme algoritmaları ve…<br>
– Güzel teorik özellikler. (Uygulaması kolay, Az bir bakım maliyeti ile bir çok problem için çözüm üretilebilir.)<br>
<br>
İstatistiksel Öğrenme Teorisi;<br>
<br>
Sistemler matematiksel olarak şu şekilde tanımlanabilir; Verileri (gözlemleri) girdi olarak alır ve gelecekteki verilerin bazı özelliklerini tahmin etmek için kullanılabilecek bir işlev çıkarır. İstatistiksel öğrenme teorisi bunu bir fonksiyon tahmin problemi olarak modeller ve genelleştirme performansı (test verilerinin etiketlenmesinde doğruluk)  olarak ölçer.
### Destek Vektör Makineleri Nedir?
Destek vektör algoritması ilk başta sınıflandırma için çıkmış bir algoritma olmasına rağmen regresyon içinde kullanılmaktadır. Bu iki model sayesinde bazı veri problemlerinin çözümü sağlanmaktadır.<br>
Elimizdeki veriler için her zaman doğrusal modeller kullanamayız. Bu gibi durumlarda başka algoritmalar ile elimizdeki verileri anlamlaştırmaya çalışırız. Bunlardan birisi de destek vektör algoritmalarıdır. Kısaca regresyon ve sınıflandırma için kullanılan iki modeli bulunmaktadır.<br>SVM için: <br>
Verileri 2 boyutlu bir düzlemde olduğu kadar çok boyutlu bir hiper düzlemde sınıflandırmak için kullanılır.<br><br>
![image](https://user-images.githubusercontent.com/106628455/204904711-e22ac5a1-5bde-42e9-a522-c8d7da192b5e.png)<br>
Grafik düzlemin görüldüğü üzere elimizde iki grup veri var ve biz bunları etiketleyerek sınıflandırma yapmak istiyoruz. Destek vektör makine algoritması uygulandığında ortadan geçen paralel çizgiyi elde etmiş oluruz. Bu sayede elimizdeki veriler ve gelecek veriler için sınıflandırma yapabilecek durumda oluruz. Grafikteki çizilen paralel aralığın kestiği noktalar destek noktalarıdır.<br>
Örnekte ortadaki çizgi, sınıfları ayıran *Decision Boundary* iken bu çizginin yanlarında bulunan kesikli-soyut çizgiler *Destek Vektörleridir*.
***Özellikler***:
<br>
-Destek Vektörler sadece bireysel gözlemin koordinatlarıdır.<br>
-Destek Vektör Makinesi, iki sınıfı (düzlem / çizgi) en iyi bir şekilde ayıran sınır çizgileridir.<br>
-Destek vektörleri karar yüzeyine (veya hiper düzleme) en yakın veri noktalarıdır.<br>
-Sınıflandırılması en zor veri noktalarıdır.<br>
-Karar yüzeyinin optimum konumu üzerinde doğrudan etkisi vardır.<br>
-Optimal hiper düzlemin en düşük “kapasiteye” sahip fonksiyon sınıfından kaynaklandığını gösterebiliriz.<br>
-Destek vektörleri hiper düzlemin en yakın veri noktalarıdır, bu sebeple bir veri kümesini bölen hiper düzlemin konumunu değiştirecek noktalardır. Bu nedenle, bir veri kümesinin kritik unsurları olarak kabul edilebilirler.<br>

### Destek Vektör Makineleri Nasıl Çalışır?
Bir düzlem üzerine yerleştirilmiş noktaları ayırmak için bir doğru çizer. Bu doğrunun, iki sınıfının noktaları için de maksimum uzaklıkta olmasını amaçlar. Karmaşık ama küçük ve orta ölçekteki veri setleri için uygundur.<br><br>
![image](https://user-images.githubusercontent.com/106628455/204907909-440fc48c-40da-4295-9db6-200decc73ce5.png)
<br><br>
Tabloda siyahlar ve beyazlar olmak üzere iki farklı sınıf var. Sınıflandırma problemlerindeki asıl amacımız gelecek verinin hangi sınıfta yer alacağını karar vermektir. Bu sınıflandırmayı yapabilmek için iki sınıfı ayıran bir doğru çizilir ve bu doğrunun ±1'i arasında kalan yeşil bölgeye Margin adı verilir. Margin ne kadar geniş ise iki veya daha fazla sınıf o kadar iyi ayrıştırılır.<br>

### DVM Avantaj ve Dezavantajları
**Avantajları**:<br>
<br>
+Yüksek boyutlu uzaylarda etkilidirler.<br>
+Boyut sayısının, örneklem sayısından fazla olduğu durumlarda etkilidirler.<br>
+Karar fonksiyonunda bir takım eğitim noktaları kullanılır. (“support vectors”). Dolayısıyla bellek verimli bir şekilde kullanılmış olur.<br>
+Çok yönlü: Karar fonksiyonu için çok farklı çekirdek fonksiyonları (“kernel functions”) kullanılabilmektedir.<br>
+Doğrusal ve Doğrusal olmayan verilere uygulanabilme<br>
+Yüksek Doğruluk oranı<br>
+Çok sayıda bağımsız değişkenle çalışabilme<br>
+Overfitting sorunun olmaması<br>
<br>
**Dezavantajları**:<br>
<br>
+Olasılıksal tahminler üretememe<br>
+Daha fazla zaman harcanır.<br>
+Büyük veri kümelerine uygun değildir.<br>
+Çakışan sınıflarla kötü çalışır.<br>
+Kullanılan çekirdek türüne duyarlıdır.<br><br>
![gif](https://miro.medium.com/max/864/0*Z1FpW3wyZjZKKrRl.gif)<br>
<br>
## DOĞRUSAL OLAN\OLMAYAN DVM
Destek vektör makineleri ikiye ayrılır:<br>
### *1.Doğrusal Destek Vektör Makineleri*<br><br>
Doğrusal olarak ayrılabilen veriler için kullanılır.
Bir veri kümesinin tek bir düz çizgi kullanılarak iki sınıfa ayrılabilmesidir.<br><br>
![image](https://user-images.githubusercontent.com/106628455/204911782-d1fd6d2c-c822-4c03-ac91-a60608a26daa.png)<br><br>
**Python ile Uygulama**<br><br>
Doğrusal SVM için basit bir python uygulaması yaparak aşağıdaki sonuçlar elde edilmiştir. Bu uygulama için “iris” verisinden yararlanılmıştır.<br><br>
![image](https://user-images.githubusercontent.com/106628455/204912217-38801e73-3ead-4e3e-bcd6-bb88d55e8880.png)
<br><br>

### *2.Doğrusal Olmayan Destek Vektör Makineleri*<br><br>
![image](https://user-images.githubusercontent.com/106628455/204913461-27783640-c7ef-4075-9f85-f1683a741d5d.png)
<br>
Doğrusal olmayan bir veri kümesinde DVM’ler doğrusal bir hiper-düzlem çizemez. Bu nedenle çekirdek numarası olarak adlandırılan **kernel trick**’ler kullanılır. Çekirdek yöntemi, doğrusal olmayan verilerde makine öğrenimini yüksek oranda arttırmaktadır.<br>
<br>
En çok kullanılan çekirdek yöntemleri:<br>
<br>
***Polynomial Kernel***<br><br>
![image](https://user-images.githubusercontent.com/106628455/204914383-8f252b61-0e64-4f39-9329-8b69251462d0.png)<br>
   Polinom çekirdeği, birden fazla dereceye sahip çekirdeklerin genel bir temsilidir. Görüntü işleme için kullanışlıdır.<br>
   'd' parametresine sahiptir. d parametresi polinom derecesini ifade eder.<br>
   Polinom çekirdeği, polinomun derecesi olan d'yi ayarlayarak boyutları sistematik olarak artırır.<br>
`K(xi,xj) = (xi.xj)d`
<br>
<br>
***Gaussian RBF (Radial Basis Function) Kernel***<br><br>
![image](https://user-images.githubusercontent.com/106628455/204915306-6d06e0b8-4cb9-4693-8f5d-bdee1dfc51f7.png)
<br>
   RBF, radyal tabanlı fonksiyondur. Veriler hakkında önceden bilgi olmadığında kullanılır.<br>
   Radyal çekirdek, sonsuz boyutlarda destek vektör sınıflandırıcısını bulur.<br>
   En yakın gözlemlerin yeni gözlemleri nasıl sınıflandıracağımız üzerinde çok fazla etkisi vardır ve daha uzaktaki gözlemlerin sınıflandırma üzerinde nispeten az etkisi vardır.<br>
`K(xi,xj) = exp(-γ||xi – xj||)2`<br>
<br>
## DVM KULLANIM ALANLARI
DVM’ler literatürde birçok örüntü uygulamasında kullanılmaktadır.
Şekilde gösterilen yüz algılama, metin ve köprü metni sınıflandırma,
görüntü sınıflandırma, biyoinformatik, protein çaprazlama, uzaktan
homoloji tespiti, el yazısı tanıma, jeoleji ve çevre bilimleri,genelleştirilmiş tahmine dayalı kontrol uygulamaları bunlardan sadece
birkaçıdır.<br><br>
![image](https://user-images.githubusercontent.com/106628455/204918512-5d53471d-df31-4c0b-80ab-475f830b5683.png)<br><br>
***1.Yüz Algılama***
Görüntünün bölümleri yüz ve yüz olmayan bölümler olarak
sınıflandırılır. NxN boyutundaki görüntüdeki her bir piksel değeri yüz
ve yüz olmayan bölümler olarak iki farklı etiketle etiketlenir. Bu veriler
eğitim verilerini oluşturur. Sonrasında piksel parlaklığına göre yüzlerin
etrafında bir karesel sınır oluşturulur ve her bir görüntü için aynı işlem
tekrar edilerek, DVM ile sınıflandırma işlemleri gerçekleştirilir.
<br><br>
***2. Metin ve Köprü Metni Sınıflandırma***
Metin ve köprü metinlerin sınıflandırılması DVM ile
gerçekleştirilebilir. Bu işlem için ilk olarak metinleri; haber makaleleri,
e-postalar ve web sayfaları gibi farklı kategorilerde sınıflandırmak için
eğitim verileri kullanılır.
Örneğin:
• Haber makalelerinin "iş" ve "filmler" olarak sınıflandırılması,
• Web sayfalarının kişisel ana sayfalar ve diğerleri olarak
sınıflandırılması.
Algoritmada sonraki adımda her belge için bir puan hesaplanır.
Hesaplanan değer önceden tanımlanmış bir eşik değeriyle karşılaştırılır.
Bir metnin puanı belirlenen eşik değerini aştığında, metin belirli bir
kategoride sınıflandırılır. Eşik değerini geçmezse, genel bir metin
olarak değerlendirir. Her metin için puan hesaplanarak ve öğrenilen
eşikle karşılaştırılarak yeni örneklerin sınıflandırılması gerçekleştirilir.
<br><br>
***3. Görüntülerin Sınıflandırılması***
DVM'ler, görüntüleri daha yüksek arama doğruluğu ile sınıflandırabilir.
Doğruluğu, geleneksel sorgu tabanlı ayrıntılandırma şemalarından daha
yüksektir.<br><br>
***4. Biyoinformatik***
Hesaplamalı biyoloji alanında, protein homoloji tespiti yaygın bir
problemdir. Bu problemi çözmek için kullanılan en etkili yöntemlerden
birisi de DVM yöntemidir. Bu yöntem, biyolojik diziler arasında
tanımlama yapmak için yaygın olarak kullanılmaktadır. Örneğin;
hastaların genlerine göre ve diğer birçok biyolojik problemlerine göre
sınıflandırılması için kullanılır.<br><br>
***5. Protein Çaprazlama ve Uzaktan Homoloji Tespiti***
Protein uzaktan homoloji tespiti, protein yapıları ve fonksiyonları
çalışmalarında hayati bir rol oynar. Neredeyse esnek hesaplama
algoritmalarının tamamı, protein dizilerini temsil etmek için sabit
uzunluk özelliklerine ihtiyaç duyar. Bununla birlikte, sınırlı protein
bilgisi ile ayırt edici özellikleri tespit etmek kolay bir iş değildir. Buna
karşın literatürde bu işlemlerde DVM’ler oldukça başarılı sonuçlar
vermektedir.<br><br>
***6. El Yazısı Tanıma***
Literatürde belgeler üzerindeki imzaları ve elle yazılmış karakterleri
tanımak için DVM yöntemleri kullanılmaktadır.<br><br>
***7. Jeoleji ve Çevre Bilimleri***
DVM'ler coğrafi (mekansal) ve mekansal-zamansal çevresel veri
analizi ve modelleme işlemleri içinde kullanılmaktadır. Özellikle uydu
görüntüleri üzerinden haritalama uygulamalarında da kullanılmaktadır.
Bu tarz uygulamalarda farklı çekirdek fonksiyonlarının başarısı üzerine
de yapılmış akademik çalışmalar bulunmaktadır.<br><br>
***8. Genelleştirilmiş Tahmine Dayalı Kontrol (GTK)***
Kaotik dinamikleri kullanışlı parametrelerle kontrol etmek için DVM
tabanlı GTK kullanılır. Bu yöntem sistem kontrolünde çok iyi derecede
performans sağlar. Sistem, hedefin yerel istikrarına göre kaotik
dinamikleri takip eder.
Kaotik sistemleri kontrol etmek için DVM'lerin kullanılmasının
avantajları:
• Kaotik bir sistemi hedefe yönlendirmek için nispeten küçük
parametre algoritmalarının kullanılmasına izin verir.
• Kaotik sistemler için bekleme süresini azaltır.
• Sistemlerin performansını korur. tahmine dayalı kontrol uygulamaları bunlardan sadece
birkaçıdır.<br><br><br>
## KAYNAKÇA<br><br>
https://data-flair.training/blogs/applications-of-svm/<br>
https://techvidvan.com/tutorials/svm-applications/<br>
https://www.kaggle.com/code/hasansezertaan/machine-learning-dersleri-s-n-fland-rma#S%C4%B1n%C4%B1fland%C4%B1rma-Nedir<br>
https://bilgisayarkavramlari.com/2013/03/31/siniflandirma-classification/<br>
https://blog.turhost.com/makine-ogrenmesi-machine-learning-nedir/<br>
https://www.embs.org/tbme/articles/a-machine-learning-shock-decision-algorithm-for-use-during-piston-driven-chest-compressions/<br>
https://linguisticmaz.medium.com/support-vector-machines-explained-ii-f2688fbf02ae<br>
https://medium.com/@k.ulgen90/makine-%C3%B6%C4%9Frenimi-b%C3%B6l%C3%BCm-4-destek-vekt%C3%B6r-makineleri-2f8010824054<br>
https://yigitsener.medium.com/destek-vekt%C3%B6r-makineleri-support-vector-machine-svm-%C3%A7al%C4%B1%C5%9Fma-mant%C4%B1%C4%9F%C4%B1-ve-python-uygulamas%C4%B1-992163ff3eec<br>
https://medium.com/deep-learning-turkiye/nedir-bu-destek-vekt%C3%B6r-makineleri-makine-%C3%B6%C4%9Frenmesi-serisi-2-94e576e4223e<br>























